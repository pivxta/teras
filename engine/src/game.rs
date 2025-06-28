use arrayvec::ArrayVec;
use dama::{Color, File, Move, MoveKind, Piece, Position, Square, ToMove};
use crate::nnue;

#[derive(Clone, Debug)]
pub struct Game {
    positions: Vec<Position>,
    moves: Vec<Option<Move>>,
    accumulators: Vec<nnue::Accumulator>,
    transformer: Option<nnue::Transformer>,
}

impl Default for Game {
    #[inline]
    fn default() -> Self {
        Self::from_initial()
    }
}

impl Game {
    pub fn from_initial() -> Self {
        Self::from_position(Position::new_initial())
    }

    pub fn from_position(position: Position) -> Self {
        Self {
            positions: vec![position],
            accumulators: Vec::new(),
            moves: Vec::new(),
            transformer: None,
        }
    }

    pub fn with_moves<M>(mut self, moves: &[M]) -> Result<Self, M::Error> 
    where 
        M: ToMove
    {
        for mv in moves {
            self.play(&mv.to_move(self.position())?);
        }
        Ok(self)
    }

    pub(crate) fn with_nnue(mut self, transformer: &nnue::Transformer) -> Self {
        self.transformer = Some(transformer.clone());
        self.accumulators.push(nnue::Accumulator::from_position(transformer, self.position()));
        self
    }

    #[inline]
    pub fn accumulator(&self) -> Option<&nnue::Accumulator> {
        self.accumulators.last()
    }

    #[inline]
    pub fn position(&self) -> &Position {
        &self.positions[self.positions.len() - 1]
    }

    #[inline]
    pub fn previous_position(&self) -> Option<&Position> {
        if self.positions.len() >= 2 {
            Some(&self.positions[self.positions.len() - 2])
        } else {
            None
        }
    }

    #[inline]
    pub fn previous_move(&self) -> Option<Move> {
        self.moves.last().and_then(|m| *m)
    }

    #[inline]
    pub fn visit_skip<T>(&mut self, mut f: impl FnMut(&mut Game) -> T) -> T {
        self.skip();
        let value = f(self);
        self.undo();
        value 
    }

    #[inline]
    pub fn visit<T>(&mut self, mv: &Move, mut f: impl FnMut(&mut Game) -> T) -> T {
        self.play(mv);
        let value = f(self);
        self.undo();
        value 
    }

    #[inline]
    fn play(&mut self, mv: &Move) {
        if let Some(transformer) = &self.transformer {
            self.accumulators.push(
                self.accumulators
                    .last()
                    .expect("NNUE position has no accumulators.")
                    .clone(),
            );
            let updates = FeatureUpdates::from_move(self.position(), mv);
            let accumulator = self.accumulators.last_mut().unwrap();
            for added in updates.added {
                accumulator.add(transformer, added.color, added.piece, added.square);
            }
            for removed in updates.removed {
                accumulator.sub(transformer, removed.color, removed.piece, removed.square);
            }
        }

        self.moves.push(Some(*mv));
        self.positions.push(self.positions.last().unwrap().clone());
        self.positions
            .last_mut()
            .unwrap()
            .play_unchecked(mv);
    }

    #[inline]
    fn skip(&mut self) {
        if self.transformer.is_some() {
            self.accumulators.push(
                self.accumulators
                    .last()
                    .expect("NNUE position has no accumulators.")
                    .clone(),
            );
        }

        self.moves.push(None);
        self.positions.push(self.positions.last().unwrap().clone());
        self.positions.last_mut().unwrap().skip();
    }

    #[inline]
    pub fn undo(&mut self) {
        if self.positions.len() > 1 {
            self.positions.pop();
        }
        self.accumulators.pop();
        self.moves.pop();
    }

    #[inline]
    pub fn is_draw(&self) -> bool {
        self.position().halfmove_clock() >= 100 || self.repetitions() >= 2
    }

    #[inline]
    pub fn repetitions(&self) -> u32 {
        let since_irreversible = self.position().halfmove_clock() as usize + 1;
        let hash = self.position().hash();
        self.positions
            .iter()
            .rev()
            .take(since_irreversible)
            .step_by(4)
            .filter(|e| e.hash() == hash)
            .count() as u32
    }
}

#[derive(Clone, Debug, Default)]
struct FeatureUpdates {
    added: ArrayVec<FeatureUpdate, 2>,
    removed: ArrayVec<FeatureUpdate, 2>,
}

impl FeatureUpdates {
    #[inline]
    fn from_move(position: &Position, mv: &Move) -> FeatureUpdates {
        let mut updates = FeatureUpdates::default();
        let us = position.side_to_move();
        let them = !position.side_to_move();

        match mv.kind {
            MoveKind::Normal { promotion } => {
                let moved = position.piece_at(mv.from).expect("no piece to be moved");
                let captured = position.piece_at(mv.to);
                if let Some(captured) = captured {
                    updates.remove(them, captured, mv.to);
                }

                updates.remove(us, moved, mv.from);
                if let Some(promotion) = promotion {
                    updates.add(us, promotion, mv.to);
                } else {
                    updates.add(us, moved, mv.to);
                }
            }
            MoveKind::Castles { rook } => {
                updates.remove(us, Piece::King, mv.from);
                updates.remove(us, Piece::Rook, rook);
                updates.add(us, Piece::King, mv.to);

                if mv.to.file() > mv.from.file() {
                    updates.add(us, Piece::Rook, rook.with_file(File::F));
                } else {
                    updates.add(us, Piece::Rook, rook.with_file(File::D));
                }
            }
            MoveKind::EnPassant { target } => {
                updates.remove(us, Piece::Pawn, mv.from);
                updates.remove(them, Piece::Pawn, target);
                updates.add(us, Piece::Pawn, mv.to);
            }
        }

        updates
    }

    #[inline]
    fn add(&mut self, color: Color, piece: Piece, square: Square) {
        self.added.push(FeatureUpdate {
            color,
            piece,
            square,
        });
    }

    #[inline]
    fn remove(&mut self, color: Color, piece: Piece, square: Square) {
        self.removed.push(FeatureUpdate {
            color,
            piece,
            square,
        });
    }
}

#[derive(Copy, Clone, Debug)]
struct FeatureUpdate {
    color: Color,
    piece: Piece,
    square: Square,
}

#[cfg(test)]
mod tests {
    use crate::nnue;
    use dama::{SanMove, ToMove, UciMove};
    use super::Game;

    impl Game {
        #[inline]
        #[cfg(test)]
        pub fn play_checked<M: ToMove>(&mut self, mv: &M) -> Result<(), M::Error> {
            self.play(&mv.to_move(self.position())?);
            Ok(())
        }
    }

    #[test]
    fn bongcloud_repetition() {
        let mut game = Game::from_initial();

        game.play_checked(&"e4".parse::<SanMove>().unwrap())
            .unwrap();
        game.play_checked(&"e5".parse::<SanMove>().unwrap())
            .unwrap();
        game.play_checked(&"Ke2".parse::<SanMove>().unwrap())
            .unwrap();
        game.play_checked(&"Ke7".parse::<SanMove>().unwrap())
            .unwrap();
        game.play_checked(&"Ke1".parse::<SanMove>().unwrap())
            .unwrap();
        game.play_checked(&"Ke8".parse::<SanMove>().unwrap())
            .unwrap();
        game.play_checked(&"Ke2".parse::<SanMove>().unwrap())
            .unwrap();
        game.play_checked(&"Ke7".parse::<SanMove>().unwrap())
            .unwrap();
        game.play_checked(&"Ke1".parse::<SanMove>().unwrap())
            .unwrap();
        game.play_checked(&"Ke8".parse::<SanMove>().unwrap())
            .unwrap();
        assert_eq!(game.repetitions(), 2);

        game.play_checked(&"Ke2".parse::<SanMove>().unwrap())
            .unwrap();
        game.play_checked(&"Ke7".parse::<SanMove>().unwrap())
            .unwrap();
        assert_eq!(game.repetitions(), 3);
    }

    #[test]
    fn horsejump_repetitions() {
        let mut game = Game::from_initial();

        game.play_checked(&"Nf3".parse::<SanMove>().unwrap())
            .unwrap();
        game.play_checked(&"Nf6".parse::<SanMove>().unwrap())
            .unwrap();
        game.play_checked(&"Ng1".parse::<SanMove>().unwrap())
            .unwrap();
        game.play_checked(&"Ng8".parse::<SanMove>().unwrap())
            .unwrap();
        assert_eq!(game.repetitions(), 2);

        let mut game = Game::from_initial();

        game.play_checked(&"Nf3".parse::<SanMove>().unwrap())
            .unwrap();
        game.play_checked(&"Nf6".parse::<SanMove>().unwrap())
            .unwrap();
        game.play_checked(&"Ng5".parse::<SanMove>().unwrap())
            .unwrap();
        game.play_checked(&"Ng4".parse::<SanMove>().unwrap())
            .unwrap();
        game.play_checked(&"Nh3".parse::<SanMove>().unwrap())
            .unwrap();
        game.play_checked(&"Nh6".parse::<SanMove>().unwrap())
            .unwrap();
        game.play_checked(&"Ng1".parse::<SanMove>().unwrap())
            .unwrap();
        game.play_checked(&"Ng8".parse::<SanMove>().unwrap())
            .unwrap();

        assert_eq!(game.repetitions(), 2);
    }

    #[test]
    fn nnue_accumulator() {
        let moves = [
            "g1f3", "d7d5", "g2g3", "c7c5", "f1g2", "b8c6", "d2d4", "e7e6", "e1g1", "c5d4", "f3d4",
            "g8e7", "c2c4", "c6d4", "d1d4", "e7c6", "d4d1", "d5d4", "e2e3", "f8c5", "e3d4", "c5d4",
            "b1c3", "e8g8", "c3b5", "d4b6", "b2b3", "a7a6", "b5c3", "b6d4", "c1b2", "e6e5", "d1d2",
            "c8e6", "c3d5", "b7b5", "c4b5", "a6b5", "d5f4", "e5f4", "g2c6", "d4b2", "d2b2", "a8b8",
            "f1d1", "d8b6", "c6f3", "f4g3", "h2g3", "b5b4", "a2a4", "b4a3", "a1a3", "g7g6", "b2d4",
            "b6b5", "b3b4", "b5b4", "d4b4", "b8b4", "a3a8", "f8a8", "f3a8", "g6g5", "a8d5", "e6f5",
            "d1c1", "g8g7", "c1c7", "f5g6", "c7c4", "b4b1", "g1g2", "b1e1", "c4b4", "h7h5", "b4a4",
            "e1e5", "d5f3", "g7h6", "g2g1", "e5e6", "a4c4", "g5g4", "f3d5", "e6d6", "d5b7", "h6g5",
            "f2f3", "f7f5", "f3g4", "h5g4", "c4b4", "g6f7", "g1f2", "d6d2", "f2g1", "g5f6", "b4b6",
            "f6g5", "b6b4", "f7e6", "b4a4", "d2b2", "b7a8", "g5f6", "a4f4", "f6e5", "f4f2", "b2f2",
            "g1f2", "e6d5", "a8d5", "e5d5", "f2e3", "d5e5", "e3d3", "f5f4", "d3e2", "f4f3", "e2e3",
            "e5f5", "e3f2", "f5e4", "f2f1", "f3f2", "f1f2", "e4d3", "f2f1", "d3e3", "f1g2", "e3e2",
            "g2g1", "e2f3", "g1h2", "f3f2", "h2h1", "f2g3", "h1g1", "g3h3", "g1f2", "h3h2", "f2f1",
            "g4g3", "f1e2", "g3g2", "e2d3", "g2g1q", "d3c4", "h2g3",
        ];

        let (_, transformer) = nnue::Model::load_default();
        let mut game = Game::from_initial().with_nnue(&transformer);
        for mv in moves {
            game.play_checked(&mv.parse::<UciMove>().unwrap()).unwrap();

            assert_eq!(
                game.accumulator(),
                Some(&nnue::Accumulator::from_position(
                    &transformer,
                    game.position()
                )),
                "{}\nmove: {}",
                game.position(),
                mv
            );
        }
    }
}
