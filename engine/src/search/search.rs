use crate::{eval::{Bound, Eval}, game::Game, nnue, psqt, utils::see::see};
use super::{
    cache::{self, CacheTable}, history::History, killers::Killers, line::Line, node::{Node, NodeKind}, order::{OrderedMoves, OrderingContext}, time::Time, window::Window, EvalMode, Job, OnDepthFinished, OnSearchFinished, SearchInfo, SearchJob, Shared
};
use dama::{Move, MoveList};
use std::{array, sync::Arc, time::Instant};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ThreadKind {
    Main,
    Worker,
}

pub struct Thread {
    kind: ThreadKind,

    cache: CacheTable,
    shared: Arc<Shared>,
    model: nnue::Model,
    transformer: nnue::Transformer,
    eval_mode: EvalMode,

    root_moves: MoveList,
    max_depth: u32,
    max_nodes: u64,
    time: Time,
    start: Instant,
    nodes: u64,

    killers: Killers,
    history: History,

    lmr_table: [[i32; 64]; 64]
}

#[derive(Clone, Copy, Debug)]
struct SearchInterrupted;

impl Thread {
    pub fn new(
        kind: ThreadKind, 
        shared: Arc<Shared>, 
        cache: CacheTable, 
        model: nnue::Model, 
        transformer: nnue::Transformer
    ) -> Self {
        Self {
            kind,
            cache,
            shared,
            model,
            transformer,

            root_moves: Default::default(),
            eval_mode: Default::default(),
            max_depth: Default::default(),
            max_nodes: Default::default(),
            time: Default::default(),
            nodes: Default::default(),
            start: Instant::now(),

            killers: Default::default(),
            history: Default::default(),

            lmr_table: array::from_fn(|depth| array::from_fn(|num| {
                if depth == 0 {
                    return 0;
                }

                let depth = depth as f32;
                let num = num as f32 + 1.0;
                (0.5 + depth.ln() * num.ln() / 2.25) as i32
            }))
        }
    }

    pub fn do_job(&mut self, job: Job) {
        match job {
            Job::Search(job) => self.do_search_job(*job),
            Job::Reset => self.reset(),
        }
    }

    fn do_search_job(&mut self, job: SearchJob) {
        self.nodes = 0;
        self.start = job.start;
        self.cache = job.cache;
        self.shared = job.shared;
        self.model = job.model;
        self.transformer = job.transformer;
        self.eval_mode = job.eval_mode;
        self.max_depth = job.max_depth;
        self.max_nodes = job.max_nodes;
        self.time = Time::new(job.game.position().side_to_move(), job.time);
        if job.moves_to_search.is_empty() {
            self.root_moves = job.game.position().legal_moves();
        } else {
            self.root_moves = job.moves_to_search.into_iter().collect();
        }
        self.iterative_deepening(
            job.game, 
            job.on_depth_finished, 
            job.on_search_finished
        );
    }
    
    fn reset(&mut self) {
        self.killers.clear();
        self.history.clear();
    }

    fn iterative_deepening(
        &mut self,
        game: Game,
        mut on_depth_finished: Option<OnDepthFinished>,
        on_search_finished: Option<OnSearchFinished>,
    ) {
        let mut game = match self.eval_mode {
            EvalMode::Nnue => game.with_nnue(&self.transformer),
            EvalMode::Psqt => game
        };

        let mut last_depth = 0;
        let mut last_eval = None;
        let mut last_pv = match self.root_moves.first() {
            Some(mv) => Line::from_move(*mv),
            None => Line::new(),
        };

        const ASPIRATION_WINDOW: i32 = 15;
        'iterative_deepening: for depth in 1..=self.max_depth {
            let mut delta = ASPIRATION_WINDOW;
            let mut window = if depth <= 3 {
                Window::FULL
            } else {
                Window::around(last_eval.unwrap(), delta)
            };
            loop {
                let mut pv = Line::new();
                let eval = match self.search(&mut game, Node::root(depth), window, &mut pv) {
                    Ok(current_eval) => current_eval,
                    Err(_) => break 'iterative_deepening,
                };

                delta += delta / 2;

                let bound = if eval <= window.alpha {
                    Bound::Upper
                } else if eval >= window.beta {
                    Bound::Lower
                } else {
                    Bound::Exact
                };

                if let Some(send_info) = &mut on_depth_finished {
                    send_info(game.position(), self.search_info(depth, eval, bound, &pv));
                }

                last_depth = depth;
                last_eval = Some(eval);
                last_pv = pv;

                match bound {
                    Bound::Upper => {
                        window.beta = window.alpha.average(window.beta);
                        window.alpha -= delta;
                    }
                    Bound::Lower => {
                        window.beta += delta; 
                    }
                    _ => break
                }
            }
            if self.start.elapsed() >= self.time.soft_limit {
                break;
            }
        }

        if self.kind == ThreadKind::Main {
            self.shared.stop();
        }

        if let Some(on_search_finished) = on_search_finished {
            on_search_finished(
                game.position(),
                self.search_info(
                    last_depth, 
                    last_eval.unwrap_or(Eval::ZERO), 
                    Bound::Exact, 
                    &last_pv
                ),
            );
        }
    }

    fn search_info(&self, depth: u32, eval: Eval, bound: Bound, pv: &Line) -> SearchInfo {
        let time_elapsed = self.start.elapsed();
        let nodes_searched = self.shared.nodes_searched();
        let nodes_per_sec = ((nodes_searched as u128 * 1000000) / time_elapsed.as_micros()) as u64;
        SearchInfo {
            depth,
            eval,
            pv: pv.to_vec(),
            bound,
            nodes_searched,
            nodes_per_sec,
            time_elapsed,
            hash_full_permill: self.cache.used_approx_permill(),
        }
    }

    fn search(
        &mut self,
        game: &mut Game,
        mut node: Node,
        mut window: Window,
        pv: &mut Line,
    ) -> Result<Eval, SearchInterrupted> {
        if node.is_leaf() {
            return self.quiescence_search(game, node, window, pv);
        }

        if self.should_stop() {
            return Err(SearchInterrupted);
        }

        self.add_node();
        pv.clear();

        if !node.is_root() {
            if game.is_draw() {
                return Ok(Eval::DRAW);
            }

            window.alpha = window.alpha.max(Eval::mated_in(node.ply));
            window.beta = window.beta.min(Eval::mate_in(node.ply));
            if window.alpha >= window.beta {
                return Ok(window.alpha)
            }
        }

        let in_check = game.position().is_in_check();

        let hash_entry = self.cache.load(game.position(), &node);
        let hash_move = hash_entry.and_then(|e| e.best);

        let mut static_eval = if !in_check {
            self.evaluate(game)
        } else {
            Eval::mated_in(node.ply)
        };

        if let Some(hash_entry) = hash_entry {
            static_eval = match hash_entry.bound {
                Bound::Exact => hash_entry.eval,
                Bound::Lower => hash_entry.eval.max(static_eval),
                Bound::Upper => hash_entry.eval.min(static_eval),
            };
        }
        
        if !node.is_pv() {
            if let Some(hash_entry) = hash_entry.filter(|e| e.depth >= node.depth) {
                match hash_entry.bound {
                    Bound::Exact => return Ok(hash_entry.eval),
                    Bound::Upper if hash_entry.eval >= window.beta => return Ok(hash_entry.eval),
                    Bound::Lower if hash_entry.eval <= window.alpha => return Ok(hash_entry.eval),
                    _ => {}
                }
            }

            let margin = node.depth as i32 * 100;
            if !in_check && static_eval >= window.beta + margin {
                return Ok(static_eval.average(window.beta));
            }

            if !in_check
                && node.depth >= 2 
                && node.is_cut() 
                && game.position().has_non_pawn_material(game.position().side_to_move())
            {
                let reduction = 2 + node.depth / 5;

                let eval = game.visit_skip(|game| self.search(
                    game, 
                    node.child(NodeKind::Cut).reduce(reduction), 
                    -window.null_beta(), 
                    pv
                ).map(|eval| -eval))?;

                if eval >= window.beta {
                    return Ok(eval);
                }
            }
        }

        if hash_move.is_none() && node.depth >= 7 && (node.is_pv() || node.is_cut()) {
            node.depth -= 1;
        }

        let mut sub_line = Line::new();

        let mut quiets = MoveList::new();
        let mut quiets_tried = 0;
        let mut skip_quiets = false;

        let moves = OrderedMoves::new(
            self.ordering_context(game, &node, hash_move),
            if node.is_root() {
                self.root_moves.clone()
            } else {
                game.position().legal_moves()
            }
        );

        let mut best_move = None;
        let mut best_eval = if !moves.is_empty() || in_check {
            Eval::mated_in(node.ply)
        } else {
            Eval::DRAW
        };

        let reduction_first = if node.is_root() { 3 } else { 1 };

        for (n, mv) in moves.enumerate() {
            let is_quiet = game.position().is_quiet(&mv);

            if is_quiet && skip_quiets {
                continue;
            }

            if !node.is_pv()
                && node.depth <= 2
                && !skip_quiets
                && is_quiet
                && quiets_tried >= 5 + node.depth * node.depth 
            {
                skip_quiets = true;
                continue;    
            }

            let mut child_node = match node.kind { 
                NodeKind::Pv if n == 0 => NodeKind::Pv,
                NodeKind::Cut if n == 0 => NodeKind::All,
                _ => NodeKind::Cut,
            };
            let child_window = match child_node {
                NodeKind::Pv => window,
                _ => window.null_alpha()
            };
            
            let reduction = if n >= reduction_first && node.depth >= 3 {
                let mut reduction = self.lmr(&node, n);
                if node.is_pv() {
                    reduction -= 1;
                }
                if node.is_cut() {
                    reduction += 1;
                }
                if self.killers.get(&node).contains(&mv) {
                    reduction -= 1;
                }
                reduction.max(0) as u32
            } else {
                0
            };

            let eval = game.visit(&mv, |game| {
                let gives_check = game.position().is_in_check();
                let extension = if gives_check {
                    1
                } else {
                    0
                };

                let mut eval = -self.search(
                    game, 
                    node.child(child_node).extend(extension).reduce(reduction), 
                    -child_window, 
                    &mut sub_line
                )?;

                if reduction > 0 && eval > window.alpha {
                    eval = -self.search(
                        game, 
                        node.child(child_node).extend(extension), 
                        -child_window, 
                        &mut sub_line
                    )?;
                }

                if node.is_pv() && !child_node.is_pv() && eval > window.alpha {
                    child_node = NodeKind::Pv;
                    eval = -self.search(
                        game, 
                        node.child(child_node).extend(extension), 
                        -window, 
                        &mut sub_line
                    )?;
                }

                Ok(eval)
            })?;

            if eval > best_eval {
                best_move = Some(mv);
                best_eval = eval;
                if (node.is_root() || best_eval > window.alpha) && node.is_pv() && child_node.is_pv() {
                    pv.set(mv, &sub_line);
                }
                if best_eval > window.alpha {
                    window.alpha = best_eval;
                }
                if best_eval >= window.beta {
                    if game.position().is_quiet(&mv) {
                        self.killers.add(&node, mv);
                        self.history.bonus(game.position(), &node, &mv);
                        for quiet in quiets {
                            self.history.penalty(game.position(), &node, &quiet);
                        }
                    }
                    break;
                }
            }

            if is_quiet {
                quiets.push(mv);
                quiets_tried += 1;
            }
        }

        self.cache.store(
            game.position(),
            &node,
            cache::Entry {
                depth: node.depth,
                best: best_move,
                eval: best_eval,
                bound: if best_eval <= window.alpha {
                    Bound::Upper
                } else if best_eval >= window.beta {
                    Bound::Lower
                } else {
                    Bound::Exact
                },
            },
        );

        Ok(best_eval)
    }

    fn quiescence_search(
        &mut self,
        game: &mut Game,
        node: Node,
        mut window: Window,
        pv: &mut Line,
    ) -> Result<Eval, SearchInterrupted> {
        if self.should_stop() {
            return Err(SearchInterrupted);
        }

        self.add_node();
        pv.clear();

        if game.is_draw() {
            return Ok(Eval::DRAW);
        }

        let in_check = game.position().is_in_check();
        let mut best_eval = if !in_check {
            self.evaluate(game)
        } else {
            Eval::mated_in(node.ply)
        };

        if best_eval > window.alpha {
            window.alpha = best_eval;
        }
        if best_eval > window.beta {
            return Ok(best_eval);
        }

        let mut sub_line = Line::new();
        //let hash_entry = self.cache.load(game.position(), &node);
        //let hash_move = hash_entry.and_then(|e| e.best);
        //let mut best_move = None;

        let moves = if !in_check {
            game.position().legal_captures()
        } else {
            game.position().legal_moves()
        };

        for mv in OrderedMoves::new(self.ordering_context(game, &node, None), moves) {
            if !in_check && see(game.position(), &mv) < 0 {
                continue;
            }

            let eval = game.visit(&mv, |game| {
                self.search(game, node.child(NodeKind::Pv), -window, &mut sub_line)
                    .map(|eval| -eval)
            })?;

            if eval > best_eval {
                best_eval = eval;
                //best_move = Some(mv);
                pv.set(mv, &sub_line);
                if best_eval > window.alpha {
                    window.alpha = best_eval;
                }
                if best_eval >= window.beta {
                    break;
                }
            }
        }

        /*
        self.cache.store(
            game.position(),
            &node,
            cache::Entry {
                depth: 0,
                best: best_move,
                eval: best_eval,
                bound: if best_eval <= window.alpha {
                    Bound::Upper
                } else if best_eval >= window.beta {
                    Bound::Lower
                } else {
                    Bound::Exact
                },
            },
        );
*/

        Ok(best_eval)
    }

    fn evaluate(&mut self, game: &Game) -> Eval {
        match self.eval_mode {
            EvalMode::Nnue => self.model.evaluate(
                game.position(),
                game
                    .accumulator()
                    .expect("NNUE position has no accumulator"),
            ),
            EvalMode::Psqt => psqt::evaluate(game.position()),
        }
    }

    fn lmr(&self, node: &Node, n: usize) -> i32 {
        self.lmr_table[node.depth.min(63) as usize][n.min(63)]
    }

    fn ordering_context<'a>(
        &self,
        game: &'a Game,
        node: &Node,
        hash_move: Option<Move>,
    ) -> OrderingContext<'a, '_> {
        OrderingContext {
            game,
            hash_move,
            history: &self.history,
            killers: self.killers.get(node),
        }
    }

    fn add_node(&mut self) {
        self.nodes = 0;
        self.shared.add_node();
    }

    fn should_stop(&self) -> bool {
        if self.shared.should_stop() {
            return true;
        }

        const NODES_PER_TIME_CHECK: u64 = 128;
        if self.kind == ThreadKind::Main
            && self.nodes % NODES_PER_TIME_CHECK == 0
            && (self.shared.nodes_searched() >= self.max_nodes || self.time_is_over())
        {
            self.shared.stop();
            return true;
        }

        false
    }

    fn time_is_over(&self) -> bool {
        self.kind == ThreadKind::Main && self.start.elapsed() >= self.time.hard_limit
    }
}
