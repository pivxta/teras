mod cache;
mod history;
mod killers;
mod line;
mod node;
mod order;
mod time;
mod window;
mod params;
mod search;

use crate::{eval::{Bound, Eval}, game::Game, nnue, psqt};
use cache::CacheTable;
use dama::{ByColor, Move, Position};
use node::Node;
use search::{Thread, ThreadKind};
use std::{
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering}, mpsc, Arc,
    }, thread::{self, JoinHandle}, time::{Duration, Instant}
};
use thiserror::Error;

pub const MAX_PLY: u32 = 255;
pub const MAX_DEPTH: u32 = MAX_PLY;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EngineOptions {
    pub cache_size_in_mb: usize,
    pub eval_mode: EvalMode,
    pub thread_count: u32,
}

impl Default for EngineOptions {
    fn default() -> Self {
        EngineOptions {
            cache_size_in_mb: 16,
            eval_mode: EvalMode::default(),
            thread_count: 1,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum TimeControl {
    #[default]
    Infinite,
    Ponder,
    MoveTime(Duration),
    Clock {
        time: ByColor<Duration>,
        increment: ByColor<Duration>,
        moves_to_go: Option<u32>,
    },
}

type OnDepthFinished = Box<dyn FnMut(&Position, SearchInfo) + Send + 'static>;
type OnSearchFinished = Box<dyn FnOnce(&Position, SearchInfo) + Send + 'static>;

#[derive(Default)]
pub struct SearchOptions {
    pub moves_to_search: Vec<Move>,
    pub depth: Option<u32>,
    pub mate: Option<u32>,
    pub nodes: Option<u64>,
    pub time: TimeControl,
    on_depth_finished: Option<OnDepthFinished>,
    on_search_finished: Option<OnSearchFinished>,
}

#[derive(Clone, Debug)]
pub struct SearchInfo {
    pub depth: u32,
    pub eval: Eval,
    pub bound: Bound,
    pub time_elapsed: Duration,
    pub nodes_searched: u64,
    pub nodes_per_sec: u64,
    pub hash_full_permill: u32,
    pub pv: Vec<Move>,
}

impl SearchOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_moves_to_search(mut self, moves_to_search: Vec<Move>) -> Self {
        self.moves_to_search = moves_to_search;
        self
    }

    pub fn with_depth(mut self, depth: u32) -> Self {
        self.depth = Some(depth);
        self
    }

    pub fn with_mate(mut self, mate: u32) -> Self {
        self.mate = Some(mate);
        self
    }

    pub fn with_nodes(mut self, nodes: u64) -> Self {
        self.nodes = Some(nodes);
        self
    }

    pub fn with_time_control(mut self, tc: TimeControl) -> Self {
        self.time = tc;
        self
    }

    pub fn on_depth_finished<F>(mut self, f: F) -> Self
    where
        F: FnMut(&Position, SearchInfo) + Send + 'static,
    {
        self.on_depth_finished = Some(Box::new(f));
        self
    }

    pub fn on_search_finished<F>(mut self, f: F) -> Self
    where
        F: FnOnce(&Position, SearchInfo) + Send + 'static,
    {
        self.on_search_finished = Some(Box::new(f));
        self
    }

    pub fn set_on_depth_finished<F>(&mut self, f: F)
    where
        F: FnMut(&Position, SearchInfo) + Send + 'static,
    {
        self.on_depth_finished = Some(Box::new(f));
    }

    pub fn set_on_search_finished<F>(&mut self, f: F)
    where
        F: FnOnce(&Position, SearchInfo) + Send + 'static,
    {
        self.on_search_finished = Some(Box::new(f));
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum Error {
    #[error("thread count cannot be 0, expected value >=1")]
    InvalidThreadCount,
    #[error("hash size cannot be 0")]
    InvalidCacheSize,
    #[error("engine is already busy")]
    Busy
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum EvalMode {
    #[default]
    Nnue,
    Psqt,
}

#[derive(Debug)]
pub struct Engine {
    shared: Arc<Shared>,
    cache: CacheTable,
    model: nnue::Model,
    transformer: nnue::Transformer,
    threads: Vec<(JoinHandle<()>, mpsc::Sender<Job>)>,
    eval_mode: EvalMode,
}

#[derive(Debug)]
struct Shared {
    stop: AtomicBool,
    nodes: AtomicU64,
}

impl Default for Shared {
    fn default() -> Self {
        Shared {
            stop: AtomicBool::new(true),
            nodes: AtomicU64::new(0),
        }
    }
}

enum Job {
    Reset,
    Search(Box<SearchJob>),
}

struct SearchJob {
    game: Game,
    cache: CacheTable,
    shared: Arc<Shared>,

    model: nnue::Model,
    transformer: nnue::Transformer,
    eval_mode: EvalMode,

    moves_to_search: Vec<Move>,
    max_depth: u32,
    max_nodes: u64,
    time: TimeControl,
    start: Instant,

    on_depth_finished: Option<OnDepthFinished>,
    on_search_finished: Option<OnSearchFinished>,
}

impl Engine {
    pub fn new(options: EngineOptions) -> Engine {
        let (model, transformer) = nnue::Model::load_default();
        let cache = CacheTable::with_size_in_mb(options.cache_size_in_mb);
        let shared = Arc::new(Shared::default());
        Engine {
            threads: (0..options.thread_count)
                .map(|n| {
                    let (job_sender, job_recv) = mpsc::channel();
                    let kind = if n == 0 {
                        ThreadKind::Main
                    } else {
                        ThreadKind::Worker
                    };
                    let cache = cache.clone();
                    let model = model.clone();
                    let transformer = transformer.clone();
                    let shared = shared.clone();

                    (
                        thread::spawn(move || {
                            search_thread(kind, cache, model, transformer, shared, job_recv)
                        }),
                        job_sender,
                    )
                })
                .collect(),
            cache,
            shared,
            model,
            transformer,
            eval_mode: options.eval_mode,
        }
    }

    pub fn model(&self) -> &nnue::Model {
        &self.model
    }

    pub fn transformer(&self) -> &nnue::Transformer {
        &self.transformer
    }

    pub fn clear_cache(&self) {
        self.cache.clear();
    }

    pub fn resize_cache(&mut self, new_size_in_mb: usize) -> Result<(), Error> {
        if new_size_in_mb == 0 {
            return Err(Error::InvalidCacheSize);
        }
        self.cache = CacheTable::with_size_in_mb(new_size_in_mb);
        Ok(())
    }

    pub fn set_thread_count(&mut self, thread_count: u32) -> Result<(), Error> {
        if thread_count < 1 {
            return Err(Error::InvalidThreadCount);
        }
        self.threads.resize_with(thread_count as usize, || {
            let (job_sender, job_recv) = mpsc::channel();
            let cache = self.cache.clone();
            let model = self.model.clone();
            let transformer = self.transformer.clone();
            let shared = self.shared.clone();
            (
                thread::spawn(move || {
                    search_thread(
                        ThreadKind::Worker,
                        cache,
                        model,
                        transformer,
                        shared,
                        job_recv,
                    )
                }),
                job_sender,
            )
        });
        Ok(())
    }

    pub fn set_eval_mode(&mut self, eval_mode: EvalMode) {
        self.eval_mode = eval_mode;
    }

    pub fn eval(&self, position: &Position, eval_mode: EvalMode) -> Eval {
        match eval_mode {
            EvalMode::Psqt => psqt::evaluate(position),
            EvalMode::Nnue => {
                let accumulator = nnue::Accumulator::from_position(self.transformer(), position);
                self.model().evaluate(position, &accumulator)
            }
        }
    }

    pub fn reset(&self) -> Result<(), Error> {
        if self.shared.running() {
            eprintln!("search call");
            return Err(Error::Busy);
        }

        for (_, jobs) in &self.threads {
            jobs.send(Job::Reset)
                .expect("search thread terminated before expected");
        }

        Ok(())
    }

    pub fn search(&mut self, game: &Game, mut options: SearchOptions) -> Result<(), Error> {
        if self.shared.running() {
            return Err(Error::Busy);
        }
        self.shared.start();
        self.cache.age();

        let mut moves_to_search = options.moves_to_search;
        moves_to_search.retain(|mv| game.position().is_legal(mv));

        for (_, jobs) in &self.threads {
            jobs.send(Job::Search(Box::new(SearchJob {
                game: game.clone(),
                cache: self.cache.clone(),
                shared: self.shared.clone(),
                model: self.model.clone(),
                transformer: self.transformer.clone(),
                eval_mode: self.eval_mode,

                moves_to_search: moves_to_search.clone(),
                max_depth: options.depth.unwrap_or(MAX_DEPTH),
                max_nodes: options.nodes.unwrap_or(u64::MAX),
                time: options.time,
                start: Instant::now(),

                on_depth_finished: options.on_depth_finished.take(),
                on_search_finished: options.on_search_finished.take(),
            })))
            .expect("search thread terminated before expected");
        }

        Ok(())
    }

    pub fn stop(&mut self) {
        self.shared.stop();
    }
}

impl Shared {
    fn start(&self) {
        self.stop.store(false, Ordering::Relaxed);
        self.nodes.store(0, Ordering::Relaxed);
    }

    fn stop(&self) {
        self.stop.store(true, Ordering::Relaxed);
    }

    fn should_stop(&self) -> bool {
        self.stop.load(Ordering::Relaxed)
    }

    fn running(&self) -> bool {
        !self.stop.load(Ordering::Relaxed)
    }

    fn add_node(&self) {
        self.nodes.fetch_add(1, Ordering::Relaxed);
    }

    fn nodes_searched(&self) -> u64 {
        self.nodes.load(Ordering::Relaxed)
    }
}

fn search_thread(
    kind: ThreadKind,
    cache: CacheTable,
    model: nnue::Model,
    transformer: nnue::Transformer,
    shared: Arc<Shared>,
    jobs: mpsc::Receiver<Job>,
) {
    let mut thread = Thread::new(kind, shared, cache, model, transformer);
    for job in jobs {
        thread.do_job(job);
    }
}
