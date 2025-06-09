use arrayvec::ArrayVec;
use dama::{MAX_LEGAL_MOVES, Move, MoveList, Position};
use crate::utils::{see::see, values};
use super::history::Histories;

pub struct OrderedMoves {
    current: usize,
    moves: MoveList,
    scores: ArrayVec<i32, MAX_LEGAL_MOVES>,
}

pub struct OrderingContext<'a, 'b> {
    pub position: &'a Position,
    pub killers: &'b [Move],
    pub histories: &'b Histories,
    pub hash_move: Option<Move>,
}

const HASH_MOVE: i32 = 10000000;
const GOOD_CAPTURE_OR_PROMOTION: i32 = 9000000;
const KILLER_MOVE: i32 = 8000000;
const QUIET_MOVE: i32 = 7000000;
const BAD_CAPTURE: i32 = 6000000;

impl OrderedMoves {
    #[inline]
    pub fn new(ctx: OrderingContext, moves: MoveList) -> Self {
        OrderedMoves {
            current: 0,
            scores: moves.iter().map(|&mv| Self::score(&ctx, mv)).collect(),
            moves,
        }
    }

    #[inline]
    fn score(ctx: &OrderingContext, mv: Move) -> i32 {
        if ctx.hash_move == Some(mv) {
            HASH_MOVE
        } else if ctx.position.is_capture(&mv) || mv.promotion().is_some() {
            let moved = ctx
                .position
                .piece_at(mv.from)
                .expect("No piece to be moved");
            let captured = ctx.position.piece_at(mv.to);
            let promotion = mv.promotion();
            if promotion.is_none() {
                let see = see(&ctx.position, &mv);
                if see >= 0 {
                    GOOD_CAPTURE_OR_PROMOTION 
                        + captured.map(values::piece).unwrap_or(0) * 10
                        - values::piece(moved)
                } else {
                    BAD_CAPTURE + see
                }
            } else {
                GOOD_CAPTURE_OR_PROMOTION
                    + promotion.map(values::piece).unwrap_or(0) * 10
                    + captured.map(values::piece).unwrap_or(0) * 10
                    - values::piece(moved)
            }
        } else if ctx.killers.contains(&mv) {
            KILLER_MOVE
        } else {
            QUIET_MOVE + ctx.histories.get(&ctx.position, &mv)
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.moves.is_empty()
    }
}

impl ExactSizeIterator for OrderedMoves {
    #[inline]
    fn len(&self) -> usize {
        self.moves.len()
    }
}

impl Iterator for OrderedMoves {
    type Item = Move;

    #[inline]
    fn next(&mut self) -> Option<Move> {
        self.scores
            .iter()
            .enumerate()
            .skip(self.current)
            .max_by(|(_, s1), (_, s2)| s1.cmp(s2))
            .map(|(n, _)| n)
            .map(|n| {
                let mv = self.moves[n];
                self.moves.swap(self.current, n);
                self.scores.swap(self.current, n);
                self.current += 1;
                mv
            })
    }
}
