from __future__ import annotations

from pathlib import Path


def _replace_once(path: str, old: str, new: str) -> None:
    content = Path(path).read_text(encoding="utf-8")
    count = content.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one Cannon target, found {count}")
    Path(path).write_text(content.replace(old, new, 1), encoding="utf-8")


def _patch_boltz_cannon() -> None:
    path = "src/foldjax/models/boltz2/models/triangle/triangle.py"
    old = '''    def body(lhs, rhs):
        # Transposing a tensor that is blocked over both grid axes is two
        # moves, not one: `transpose_perm` sends the tile on (i, j) to (j, i),
        # and `swapaxes` transposes each tile's own pair axes. Doing only the
        # grid half leaves every device holding the right tile indexed the
        # wrong way round, which is silently wrong rather than a shape error --
        # every tile here is square.
        if direction == "outgoing":
            rhs = jnp.swapaxes(permute(rhs, transpose_perm(side)), 1, 2)
        else:
            lhs = jnp.swapaxes(permute(lhs, transpose_perm(side)), 1, 2)
        lhs = permute(lhs, row_skew_perm(side))
        rhs = permute(rhs, col_skew_perm(side))
        total = jnp.zeros(
            lhs.shape[:2] + rhs.shape[2:3] + lhs.shape[3:], dtype=jnp.float32
        )
        for step in range(side):
            total = total + jnp.einsum(
                "bikd,bkjd->bijd", lhs, rhs, preferred_element_type=jnp.float32
            )
            if step + 1 < side:
                lhs = permute(lhs, ring_perm(side, axis=CP_COL_AXIS, delta=-1))
                rhs = permute(rhs, ring_perm(side, axis=CP_ROW_AXIS, delta=-1))
        return total
'''
    new = '''    def body(lhs, rhs):
        # Cannon transposes tile ownership onto the standard (i,k)@(k,j)
        # grid, but the resident tile keeps the serial tensor's internal axis
        # order. Using the same direction-specific einsum as the unsharded
        # program preserves its local reduction layout and avoids an otherwise
        # unnecessary swapaxes-induced fp32 drift through residual stacks.
        if direction == "outgoing":
            rhs = permute(rhs, transpose_perm(side))
            equation = "bikd,bjkd->bijd"
            rows = lhs.shape[1]
            cols = rhs.shape[1]
        else:
            lhs = permute(lhs, transpose_perm(side))
            equation = "bkid,bkjd->bijd"
            rows = lhs.shape[2]
            cols = rhs.shape[2]
        lhs = permute(lhs, row_skew_perm(side))
        rhs = permute(rhs, col_skew_perm(side))
        total = jnp.zeros(
            (lhs.shape[0], rows, cols, lhs.shape[-1]),
            dtype=jnp.float32,
        )
        correction = jnp.zeros_like(total)
        for step in range(side):
            partial = jnp.einsum(
                equation,
                lhs,
                rhs,
                preferred_element_type=jnp.float32,
            )
            updated = total + partial
            residual = jnp.where(
                jnp.abs(total) >= jnp.abs(partial),
                (total - updated) + partial,
                (partial - updated) + total,
            )
            total = updated
            correction = correction + residual
            if step + 1 < side:
                lhs = permute(lhs, ring_perm(side, axis=CP_COL_AXIS, delta=-1))
                rhs = permute(rhs, ring_perm(side, axis=CP_ROW_AXIS, delta=-1))
        return total + correction
'''
    _replace_once(path, old, new)


def _patch_protenix_cannon() -> None:
    path = "src/foldjax/models/protenix/models/triangle/triangle.py"
    old = '''        total = jnp.zeros((rows, cols) + lhs.shape[2:], dtype=jnp.float32)
        for step in range(side):
            total = total + jnp.einsum(
                equation, lhs, rhs, preferred_element_type=jnp.float32
            )
            if step + 1 < side:
                lhs = permute(lhs, ring_perm(side, axis=CP_COL_AXIS, delta=-1))
                rhs = permute(rhs, ring_perm(side, axis=CP_ROW_AXIS, delta=-1))
        return total
'''
    new = '''        total = jnp.zeros((rows, cols) + lhs.shape[2:], dtype=jnp.float32)
        correction = jnp.zeros_like(total)
        for step in range(side):
            partial = jnp.einsum(
                equation, lhs, rhs, preferred_element_type=jnp.float32
            )
            updated = total + partial
            residual = jnp.where(
                jnp.abs(total) >= jnp.abs(partial),
                (total - updated) + partial,
                (partial - updated) + total,
            )
            total = updated
            correction = correction + residual
            if step + 1 < side:
                lhs = permute(lhs, ring_perm(side, axis=CP_COL_AXIS, delta=-1))
                rhs = permute(rhs, ring_perm(side, axis=CP_ROW_AXIS, delta=-1))
        return total + correction
'''
    _replace_once(path, old, new)


def main() -> None:
    _patch_boltz_cannon()
    _patch_protenix_cannon()
    print("serial-layout compensated Cannon reduction applied")


if __name__ == "__main__":
    main()
