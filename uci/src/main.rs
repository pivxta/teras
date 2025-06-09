mod command;

use anyhow::Context;
use command::{Command, Go};
use dama::{IllegalMoveError, Position, ToMove};
use std::{io, ops::ControlFlow, time::Duration};
use teras_engine::{
    eval::{Bound, EvalKind}, game::Game, search::{Engine, EngineOptions, EvalMode, SearchInfo, SearchOptions, TimeControl}
};

fn main() {
    let mut uci = Uci::new();
    for line in io::stdin().lines() {
        let cmd = match line.expect("failed to read from stdin").parse() {
            Ok(cmd) => cmd,
            Err(error) => {
                eprintln!("error: {}", error);
                continue;
            }
        };
        match uci.run(cmd) {
            Ok(ControlFlow::Continue(_)) => continue,
            Ok(ControlFlow::Break(_)) => break,
            Err(err) => eprintln!("error: {}", err)
        }
   }
}

struct Uci {
    game: Game,
    engine: Engine,
}

impl Uci {
    fn new() -> Uci {
        Uci {
            game: Game::default(),
            engine: Engine::new(EngineOptions::default()),
        }
    }

    fn run(&mut self, command: Command) -> anyhow::Result<ControlFlow<(), ()>> {
        match command {
            Command::Uci => {
                self.id();
                self.uci_ok();
            }
            Command::IsReady => {
                self.ready_ok();
            }
            Command::SetOption { name, value } => {
                self.set_option(name.as_str(), value.as_deref())?;
            }
            Command::Position {
                position: initial_pos,
                moves,
            } => {
                self.game = Game::from_position(initial_pos).with_moves(&moves)?;
            },
            Command::Go(go) => {
                self.engine.search(&self.game, self.search_options(go)?);
            }
            Command::Stop => {
                self.engine.stop();
            }
            Command::NewGame => {
                self.engine.reset();
            }
            Command::Eval => {
                self.show_eval()?;
            }
            Command::Debug(_) => {}
            Command::PonderHit => {}
            Command::Quit => {
                return Ok(ControlFlow::Break(()));
            }
        }
        Ok(ControlFlow::Continue(()))
    }

    fn id(&self) {
        println!("id name Teras {}", env!("CARGO_PKG_VERSION"));
        println!("id author {}", env!("CARGO_PKG_AUTHORS"));
    }

    fn uci_ok(&self) {
        println!("uciok");
    }

    fn ready_ok(&self) {
        println!("readyok");
    }

    fn show_eval(&self) -> Result<(), IllegalMoveError> {
        let position = self.game.position();
        println!("Evaluation NNUE: {}", self.engine.eval(position, EvalMode::Nnue));
        println!("Evaluation PSQT: {}", self.engine.eval(position, EvalMode::Psqt));
        Ok(())
    }

    fn show_search_info(position: &Position, info: SearchInfo) {
        print!("info depth {}", info.depth);
        match info.eval.kind() {
            EvalKind::MateIn(ply) => print!(" score mate {}", ply.div_ceil(2)),
            EvalKind::MatedIn(ply) => print!(" score mate -{}", ply.div_ceil(2)),
            EvalKind::Centipawns(cp) => print!(" score cp {}", cp),
        }
        match info.bound {
            Bound::Upper => print!(" upperbound"),
            Bound::Lower => print!(" lowerbound"),
            _ => {}
        }
        print!(
            " time {} nodes {} nps {} hashfull {}",
            info.time_elapsed.as_millis(),
            info.nodes_searched,
            info.nodes_per_sec,
            info.hash_full_permill
        );
        print!(" pv");
        for mv in info.pv {
            print!(" {}", mv.to_uci(position.variant()));
        }
        println!();
    }

    fn show_search_results(position: &Position, info: SearchInfo) {
        if let Some(mv) = info.pv.first() {
            println!("bestmove {}", mv.to_uci(position.variant()));
        } else {
            println!("bestmove (none)");
        }
    }

    fn search_options(&self, go: Go) -> anyhow::Result<SearchOptions> {
        let position = self.game.position();

        let mut options = SearchOptions::default();
        options.set_on_depth_finished(Uci::show_search_info);
        options.set_on_search_finished(Uci::show_search_results);
        options.depth = go.depth;
        options.mate = go.mate;
        options.nodes = go.nodes;
        options.time = if go.ponder {
            TimeControl::Ponder
        } else if let Some(movetime) = go.move_time {
            TimeControl::MoveTime(movetime)
        } else if go.time.white.is_some() || go.time.black.is_some() {
            TimeControl::Clock {
                time: go.time.map(|time| time.unwrap_or(Duration::MAX)),
                increment: go.increment,
                moves_to_go: go.moves_to_go,
            }
        } else {
            TimeControl::Infinite
        };
        options.moves_to_search = go
            .moves_to_search
            .into_iter()
            .map(|mv| mv.to_move(&position))
            .collect::<Result<Vec<_>, _>>()?;

        Ok(options)
    }

    fn set_option(&mut self, name: &str, value: Option<&str>) -> anyhow::Result<()> {
        match name {
            "Use NNUE" if option_as_bool(value)? => self.engine.set_eval_mode(EvalMode::Nnue),
            "Use NNUE" => self.engine.set_eval_mode(EvalMode::Psqt),
            "Threads" => self.engine.set_thread_count(option_as_u32(value)?)?,
            "Hash" => self.engine.resize_cache(option_as_usize(value)?)?,
            "Clear Hash" => self.engine.reset(),
            _ => return Err(anyhow::Error::msg(format!("invalid option '{}'", name))),
        }
        Ok(())
    }
}

fn option_as_u32(value: Option<&str>) -> anyhow::Result<u32> {
    Ok(value
        .context("expected integer value")?
        .parse()
        .context("invalid value, expected integer")?)
}

fn option_as_usize(value: Option<&str>) -> anyhow::Result<usize> {
    Ok(value
        .context("expected integer value")?
        .parse()
        .context("invalid value, expected integer")?)
}

fn option_as_bool(value: Option<&str>) -> anyhow::Result<bool> {
    match value {
        Some("true") | None => Ok(true),
        Some("false") => Ok(false),
        _ => Err(anyhow::Error::msg(
            "invalid bool value, expected `true` or `false`",
        )),
    }
}
