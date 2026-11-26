use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU64, AtomicUsize, Ordering::SeqCst};
use std::sync::Arc;

/// Workers that spin between sections, the caller taking pieces too; a
/// section opened inside a piece runs on the calling thread.
pub struct Spin {
    threads: usize,
    shared: Arc<Shared>,
}

#[derive(Default)]
struct Shared {
    epoch: AtomicU64,
    job: AtomicPtr<&'static (dyn Fn(usize) + Sync)>,
    n: AtomicUsize,
    next: AtomicUsize,
    done: AtomicUsize,
    active: AtomicUsize,
    stop: AtomicBool,
    /// A section in progress: one opened meanwhile runs on its thread.
    busy: AtomicBool,
}

impl Shared {
    fn take(&self, f: &(dyn Fn(usize) + Sync)) {
        let n = self.n.load(SeqCst);
        loop {
            let i = self.next.fetch_add(1, SeqCst);
            if i >= n {
                return;
            }
            f(i);
            self.done.fetch_add(1, SeqCst);
        }
    }
}

impl Spin {
    pub fn new(threads: usize) -> Self {
        let shared = Arc::new(Shared::default());
        for _ in 1..threads {
            let sh = shared.clone();
            std::thread::spawn(move || {
                let mut seen = 0;
                while !sh.stop.load(SeqCst) {
                    let e = sh.epoch.load(SeqCst);
                    if e == seen {
                        std::hint::spin_loop();
                        continue;
                    }
                    seen = e;
                    sh.active.fetch_add(1, SeqCst);
                    let job = sh.job.load(SeqCst);
                    if !job.is_null() {
                        // SAFETY: the job lives until `active` drops to zero.
                        sh.take(unsafe { *job });
                    }
                    sh.active.fetch_sub(1, SeqCst);
                }
            });
        }
        Spin { threads, shared }
    }
}

impl rslab::parallel::Pool for Spin {
    fn threads(&self) -> usize {
        self.threads
    }
    fn run(&self, n: usize, f: &(dyn Fn(usize) + Sync)) {
        let sh = &*self.shared;
        if n < 2 || sh.busy.swap(true, SeqCst) {
            (0..n).for_each(f);
            return;
        }
        // SAFETY: `f` outlives the section; the workers drop it before return.
        let f_static: &'static (dyn Fn(usize) + Sync) = unsafe { std::mem::transmute(f) };
        let cell = f_static;
        sh.n.store(n, SeqCst);
        sh.next.store(0, SeqCst);
        sh.done.store(0, SeqCst);
        sh.job.store(&cell as *const _ as *mut _, SeqCst);
        sh.epoch.fetch_add(1, SeqCst);
        sh.take(f);
        while sh.done.load(SeqCst) < n {
            std::hint::spin_loop();
        }
        sh.job.store(std::ptr::null_mut(), SeqCst);
        while sh.active.load(SeqCst) != 0 {
            std::hint::spin_loop();
        }
        sh.busy.store(false, SeqCst);
    }
}
