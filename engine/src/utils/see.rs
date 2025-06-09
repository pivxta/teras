use super::values;
use arrayvec::ArrayVec;
use dama::{Move, Piece, Position, SquareSet, SquareSets};

pub fn see(position: &Position, mv: &Move) -> i32 {
    let target_sq = mv.to;
    let mut occupied = position.occupied().toggled(mv.from);
    let mut attackers = position.attacking_with(target_sq, occupied);

    let mut gains = ArrayVec::<i32, 64>::new();
    let mut color = !position.side_to_move();
    let mut target = position.piece_at(mv.from).expect("no piece to be moved");

    let initial_victim = match position.piece_at(target_sq) {
        Some(piece) => piece,
        None => return 0,
    };
    gains.push(values::piece(initial_victim));

    'exc: loop {
        for piece in Piece::all() {
            let piece_attackers = attackers & position.colored(color) & position.pieces(piece);
            if let Some(attacker_sq) = piece_attackers.first() {
                gains.push(values::piece(target));

                if target == Piece::King {
                    break;
                }

                occupied.toggle(attacker_sq);
                attackers.toggle(attacker_sq);
                target = piece;

                if matches!(piece, Piece::Rook | Piece::Queen) {
                    attackers |= SquareSet::rook_moves(target_sq, occupied)
                        & (position.rooks() | position.queens())
                        & occupied;
                }

                if matches!(piece, Piece::Pawn | Piece::Bishop | Piece::Queen) {
                    attackers |= SquareSet::bishop_moves(target_sq, occupied)
                        & (position.bishops() | position.queens())
                        & occupied;
                }

                color = !color;
                continue 'exc;
            }
        }
        break;
    }

    while gains.len() > 1 {
        let is_forced = gains.len() == 2;
        let their_gain = gains.pop().unwrap();
        let our_gain = gains.last_mut().unwrap();
        *our_gain -= their_gain;

        if !is_forced && *our_gain < 0 {
            *our_gain = 0;
        }
    }

    gains[0]
}

#[cfg(test)]
mod tests {
    use super::see;
    use dama::Move;
    use dama::Position;
    use dama::Square::*;

    #[test]
    fn position1() {
        let position = Position::from_fen("1k1r4/1pp4p/p7/4p3/8/P5P1/1PP4P/2K1R3 w - - ").unwrap();
        assert_eq!(see(&position, &Move::new_normal(E1, E5)), 100);
    }

    #[test]
    fn position2() {
        let position =
            Position::from_fen("1k1r3q/1ppn3p/p4b2/4p3/8/P2N2P1/1PP1R1BP/2K1Q3 w - - ").unwrap();
        assert_eq!(see(&position, &Move::new_normal(D3, E5)), 100);
    }
}
