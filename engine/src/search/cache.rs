use dama::{File, Move, MoveKind, Piece, Position, Square};
use std::{
    mem,
    sync::{
        Arc,
        atomic::{AtomicU8, AtomicU64, Ordering},
    },
};

use super::{MAX_PLY, Node};
use crate::eval::{Bound, Eval, EvalKind};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Entry {
    pub bound: Bound,
    pub eval: Eval,
    pub best: Option<Move>,
    pub depth: u32,
}

#[derive(Clone, Debug)]
pub struct CacheTable(Arc<CacheTableInner>);

impl CacheTable {
    pub fn with_size_in_mb(size_in_mb: usize) -> Self {
        Self(Arc::new(CacheTableInner::with_size_in_mb(size_in_mb)))
    }

    #[inline]
    pub fn age(&self) {
        self.0.age()
    }

    pub fn clear(&self) {
        self.0.clear();
    }

    pub fn used_approx_permill(&self) -> u32 {
        self.0.used_approx_permill()
    }

    #[inline]
    pub fn load(&self, position: &Position, node: &Node) -> Option<Entry> {
        self.0.load(position, node)
    }

    #[inline]
    pub fn store(&self, position: &Position, node: &Node, entry: Entry) {
        self.0.store(position, node, entry);
    }
}

#[derive(Debug)]
struct CacheTableInner {
    age: AtomicU8,
    entries: Box<[TableEntry]>,
}

impl CacheTableInner {
    fn with_size_in_mb(size_in_mb: usize) -> Self {
        let size_bytes = size_in_mb * 1024 * 1024;
        let entry_count = size_bytes / mem::size_of::<TableEntry>();
        Self {
            entries: (0..entry_count).map(|_| TableEntry::new()).collect(),
            age: AtomicU8::new(0),
        }
    }

    fn age(&self) {
        self.age.fetch_add(1, Ordering::Relaxed);
    }

    fn clear(&self) {
        for entry in &self.entries {
            entry.clear();
        }
    }

    fn used_approx_permill(&self) -> u32 {
        self.entries
            .iter()
            .take(1000)
            .filter(|e| !e.is_empty())
            .count() as u32
    }

    #[inline]
    fn load(&self, position: &Position, node: &Node) -> Option<Entry> {
        let hash = position.hash();
        let index = self.index(hash);
        self.entries[index].load(hash).map(|(mut entry, _)| {
            entry.eval = match entry.eval.kind() {
                EvalKind::Centipawns(cp) => Eval::centipawns(cp),
                EvalKind::MateIn(ply) => Eval::mate_in((ply + node.ply).min(MAX_PLY)),
                EvalKind::MatedIn(ply) => Eval::mated_in((ply + node.ply).min(MAX_PLY)),
            };
            entry
        })
    }

    #[inline]
    fn store(&self, position: &Position, node: &Node, mut entry: Entry) {
        entry.eval = match entry.eval.kind() {
            EvalKind::Centipawns(cp) => Eval::centipawns(cp),
            EvalKind::MateIn(ply) => Eval::mate_in(ply - node.ply),
            EvalKind::MatedIn(ply) => Eval::mated_in(ply - node.ply),
        };

        let hash = position.hash();
        let index = self.index(hash);
        let age = self.age.load(Ordering::Relaxed);

        if let Some((current_entry, current_age)) = self.entries[index].load(hash) {
            if age > current_age || entry.depth > current_entry.depth {
                self.entries[index].store(hash, entry, age);
            }
        } else {
            self.entries[index].store(hash, entry, age);
        }
    }

    #[inline]
    fn index(&self, hash: u64) -> usize {
        (hash % self.entries.len() as u64) as usize
    }
}

#[derive(Debug, Default)]
struct TableEntry {
    key: AtomicU64,
    data: AtomicU64,
}

impl TableEntry {
    #[inline]
    fn new() -> Self {
        Self::default()
    }

    #[inline]
    fn clear(&self) {
        self.data.store(0, Ordering::Relaxed);
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.data.load(Ordering::Relaxed) == 0
    }

    #[inline]
    fn load(&self, hash: u64) -> Option<(Entry, u8)> {
        let key = self.key.load(Ordering::Relaxed);
        let data = self.data.load(Ordering::Relaxed);
        if data == 0 || key ^ hash != data {
            return None;
        }
        let entry: PackedEntry = bytemuck::cast(data);
        Some(entry.unpack())
    }

    #[inline]
    fn store(&self, hash: u64, entry: Entry, age: u8) {
        let data = bytemuck::cast(entry.pack(age));
        self.data.store(data, Ordering::Relaxed);
        self.key.store(hash ^ data, Ordering::Relaxed);
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
struct PackedEntry {
    bound: u8,
    depth: u8,
    eval: PackedEval,
    mv: PackedMove,
    age: u8,
    _unused: u8,
}

impl PackedEntry {
    #[inline]
    fn unpack(&self) -> (Entry, u8) {
        (
            Entry {
                depth: self.depth as u32,
                best: if self.mv != PackedMove::null() {
                    Some(self.mv.unpack())
                } else {
                    None
                },
                bound: match self.bound {
                    0 => Bound::Exact,
                    1 => Bound::Lower,
                    2 => Bound::Upper,
                    _ => panic!("invalid transposition table entry"),
                },
                eval: self.eval.unpack(),
            },
            self.age,
        )
    }
}

impl Entry {
    #[inline]
    fn pack(&self, age: u8) -> PackedEntry {
        PackedEntry {
            bound: self.bound as u8,
            depth: self.depth as u8,
            eval: self.eval.into(),
            mv: if let Some(mv) = self.best {
                mv.into()
            } else {
                PackedMove::null()
            },
            age,
            _unused: 0,
        }
    }
}

#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
struct PackedEval(i16);

impl From<Eval> for PackedEval {
    #[inline]
    fn from(eval: Eval) -> Self {
        match eval.kind() {
            EvalKind::Centipawns(cp) => Self::centipawns(cp),
            EvalKind::MateIn(ply) => Self::mate_in(ply),
            EvalKind::MatedIn(ply) => Self::mated_in(ply),
        }
    }
}

impl PackedEval {
    const MATE_VALUE: i16 = i16::MAX;
    const MAX_CENTIPAWNS: i16 = Self::MATE_VALUE - MAX_PLY as i16 - 1;
    const MIN_CENTIPAWNS: i16 = -Self::MATE_VALUE + MAX_PLY as i16 + 1;

    #[inline]
    fn centipawns(cp: i32) -> PackedEval {
        PackedEval(cp.clamp(Self::MIN_CENTIPAWNS as i32, Self::MAX_CENTIPAWNS as i32) as i16)
    }

    #[inline]
    fn mate_in(ply: u32) -> PackedEval {
        PackedEval(Self::MATE_VALUE - ply as i16)
    }

    #[inline]
    fn mated_in(ply: u32) -> PackedEval {
        PackedEval(-Self::MATE_VALUE + ply as i16)
    }

    #[inline]
    fn plies_from_mate(self) -> Option<u32> {
        let abs = self.0.abs();
        if abs > Self::MAX_CENTIPAWNS {
            Some((Self::MATE_VALUE - abs) as u32)
        } else {
            None
        }
    }

    #[inline]
    fn unpack(self) -> Eval {
        if let Some(plies) = self.plies_from_mate() {
            if self.0 > 0 {
                return Eval::mate_in(plies);
            } else {
                return Eval::mated_in(plies);
            }
        }
        Eval::centipawns(self.0 as i32)
    }
}

#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
struct PackedMove(u16);

impl From<Move> for PackedMove {
    #[inline]
    fn from(mv: Move) -> Self {
        let extra = match mv.kind {
            MoveKind::Normal { promotion: None } => NORMAL_MOVE,
            MoveKind::Normal {
                promotion: Some(promotion),
            } => promotion_move(promotion),
            MoveKind::EnPassant { target: _ } => EN_PASSANT_MOVE,
            MoveKind::Castles { .. } if mv.to.file() == File::C => CASTLING_MOVE | QUEEN_SIDE,
            MoveKind::Castles { .. } if mv.to.file() == File::G => CASTLING_MOVE | KING_SIDE,
            MoveKind::Castles { .. } => panic!("failed to pack invalid move"),
        };
        let to = if let MoveKind::Castles { rook } = mv.kind {
            rook as u16
        } else {
            mv.to as u16
        };

        PackedMove(mv.from as u16 | (to << 6) | extra)
    }
}

impl PackedMove {
    #[inline]
    fn null() -> PackedMove {
        PackedMove(0)
    }

    #[inline]
    fn from(self) -> Square {
        unsafe { Square::from_index_unchecked((self.0 & FROM_MASK) as usize) }
    }

    #[inline]
    fn to(self) -> Square {
        let to = self.to_or_castling_rook();
        if self.0 & KIND_MASK == CASTLING_MOVE {
            let file = match self.0 & SIDE_MASK {
                QUEEN_SIDE => File::C,
                KING_SIDE => File::G,
                _ => unreachable!(),
            };
            to.with_file(file)
        } else {
            to
        }
    }

    #[inline]
    fn to_or_castling_rook(self) -> Square {
        unsafe { Square::from_index_unchecked(((self.0 & TO_MASK) >> 6) as usize) }
    }

    #[inline]
    fn kind(self) -> MoveKind {
        match self.0 & KIND_MASK {
            NORMAL_MOVE => MoveKind::Normal { promotion: None },
            PROMOTION_MOVE => MoveKind::Normal {
                promotion: Some(promotion_piece(self.0)),
            },
            CASTLING_MOVE => MoveKind::Castles {
                rook: self.to_or_castling_rook(),
            },
            EN_PASSANT_MOVE => MoveKind::EnPassant {
                target: self.to_or_castling_rook().with_rank(self.from().rank()),
            },
            _ => unreachable!(),
        }
    }

    #[inline]
    fn unpack(self) -> Move {
        Move {
            from: self.from(),
            to: self.to(),
            kind: self.kind(),
        }
    }
}

const NORMAL_MOVE: u16 = 0;
const PROMOTION_MOVE: u16 = 1 << 12;
const EN_PASSANT_MOVE: u16 = 2 << 12;
const CASTLING_MOVE: u16 = 3 << 12;

const FROM_MASK: u16 = 0x003f;
const TO_MASK: u16 = 0x0fc0;
const KIND_MASK: u16 = 0x3000;
const SIDE_MASK: u16 = 1 << 14;

const QUEEN_SIDE: u16 = 1 << 14;
const KING_SIDE: u16 = 0 << 14;

#[inline]
fn promotion_piece(mv: u16) -> Piece {
    match mv >> 14 {
        0 => Piece::Knight,
        1 => Piece::Bishop,
        2 => Piece::Rook,
        3 => Piece::Queen,
        _ => unreachable!(),
    }
}

#[inline]
fn promotion_move(piece: Piece) -> u16 {
    debug_assert_ne!(piece, Piece::Pawn);
    debug_assert_ne!(piece, Piece::King);
    PROMOTION_MOVE | ((piece as u16 - 1) << 14)
}

#[cfg(test)]
mod tests {
    use super::PackedMove;
    use dama::{Move, Piece::*, Square::*};

    #[test]
    fn packed_move_roundtrip() {
        pack_unpack(Move::new_normal(E2, E4));
        pack_unpack(Move::new_promotion(E7, E8, Queen));
        pack_unpack(Move::new_en_passant(A4, B3, B4));
        pack_unpack(Move::new_castles(E8, G8, H8));
        pack_unpack(Move::new_castles(E1, C1, A1));
        pack_unpack(Move::new_castles(E8, G8, G8));
        pack_unpack(Move::new_castles(E1, C1, D1));
    }

    fn pack_unpack(mv: Move) {
        let packed: PackedMove = mv.into();
        let unpacked = packed.unpack();
        assert_eq!(mv, unpacked);
    }
}
