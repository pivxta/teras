use std::time::Duration;

use dama::{ByColor, Color};

use super::TimeControl;

#[derive(Clone, Copy, Default, Debug)]
pub struct Time {
    pub soft_limit: Duration,
    pub hard_limit: Duration,
}

impl Time {
    pub fn new(side_to_move: Color, time: TimeControl) -> Self {
        match time {
            TimeControl::Infinite | TimeControl::Ponder => Self {
                soft_limit: Duration::MAX,
                hard_limit: Duration::MAX,
            },
            TimeControl::MoveTime(time) => Self {
                soft_limit: time,
                hard_limit: time,
            },
            TimeControl::Clock {
                time,
                increment,
                moves_to_go,
            } => Self::from_clock(side_to_move, time, increment, moves_to_go),
        }
    }

    fn from_clock(
        side_to_move: Color,
        time: ByColor<Duration>,
        inc: ByColor<Duration>,
        moves_to_go: Option<u32>,
    ) -> Self {
        let moves_to_go = moves_to_go.unwrap_or(25).min(25);
        let hard_limit = time[side_to_move] / moves_to_go + 2 * inc[side_to_move] / 3;
        let soft_limit = hard_limit / 2;
        Self {
            hard_limit,
            soft_limit,
        }
    }
}
