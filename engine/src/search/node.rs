#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum NodeKind {
    Pv,
    All,
    Cut,
}

impl NodeKind {
    #[inline]
    pub fn is_pv(self) -> bool {
        self == NodeKind::Pv
    }

    #[inline]
    pub fn is_all(self) -> bool {
        self == NodeKind::All
    }

    #[inline]
    pub fn is_cut(self) -> bool {
        self == NodeKind::Cut
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Node {
    pub kind: NodeKind,
    pub depth: u32,
    pub ply: u32,
    pub allow_null: bool,
}

impl Node {
    #[inline]
    pub const fn root(depth: u32) -> Node {
        Node {
            kind: NodeKind::Pv,
            depth,
            ply: 0,
            allow_null: true,
        }
    }

    #[inline]
    pub fn child(self, kind: NodeKind) -> Node {
        Node {
            kind,
            ply: self.ply + 1,
            allow_null: true,
            depth: self.depth.saturating_sub(1),
        }
    }

    #[inline]
    pub fn is_pv(&self) -> bool {
        self.kind == NodeKind::Pv
    }

    #[inline]
    pub fn is_cut(&self) -> bool {
        self.kind == NodeKind::Cut
    }

    #[inline]
    pub fn is_all(&self) -> bool {
        self.kind == NodeKind::All
    }

    #[inline]
    pub fn extend(mut self, extension: u32) -> Node {
        self.depth += extension;
        self
    }

    #[inline]
    pub fn reduce(mut self, reduction: u32) -> Node {
        self.depth = self.depth.saturating_sub(reduction);
        self
    }

    #[inline]
    pub fn allow_null(mut self, allow_null: bool) -> Node {
        self.allow_null = allow_null;
        self
    }

    #[inline]
    pub fn is_leaf(&self) -> bool {
        self.depth == 0
    }

    #[inline]
    pub fn is_root(&self) -> bool {
        self.ply == 0
    }
}
