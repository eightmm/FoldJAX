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
    old = '''        total = jnp.zeros(
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
    new = '''        total = jnp.zeros(
            lhs.shape[:2] + rhs.shape[2:3] + lhs.shape[3:], dtype=jnp.float32
        )
        correction = jnp.zeros_like(total)
        for step in range(side):
            partial = jnp.einsum(
                "bikd,bkjd->bijd", lhs, rhs, preferred_element_type=jnp.float32
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
    print("compensated Cannon tile reduction applied")


if __name__ == "__main__":
    main()
