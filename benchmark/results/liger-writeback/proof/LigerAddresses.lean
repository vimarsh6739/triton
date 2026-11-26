import Std

/- Liger SwiGLU at cc29eda8 uses program_id(0) for the row, program_id(1)
   for the column tile, and arange(0, block) for the logical lane. In the
   one-program-per-row kernel the tile index is zero. Both launch grids have
   a singleton third axis. The caller supplies distinct dense A/B buffers.
   This proves a property of offsets within one buffer. It assumes valid
   allocations and nonwrapping integer indexing. It does not prove alias
   freedom between arguments, warp-level read-before-overwrite ordering,
   zero-initialized gradient semantics, or floating-point equivalence. -/

namespace LigerAddresses

private theorem splitAddress {stride r s c d : Nat}
    (hc : c < stride) (hd : d < stride)
    (same : stride * r + c = stride * s + d) : r = s ∧ c = d := by
  have positive : 0 < stride := by omega
  have rows := congrArg (fun a => a / stride) same
  simp only [Nat.mul_add_div positive, Nat.div_eq_of_lt hc,
    Nat.div_eq_of_lt hd, Nat.add_zero] at rows
  subst s
  exact ⟨rfl, Nat.add_left_cancel same⟩

-- The valid-column mask and width ≤ row stride keep each tile in its row.
theorem uniqueOwner {stride width block r s tile tile' lane lane' : Nat}
    (fits : width ≤ stride) (hl : lane < block) (hl' : lane' < block)
    (valid : block * tile + lane < width)
    (valid' : block * tile' + lane' < width)
    (same : stride * r + (block * tile + lane) =
            stride * s + (block * tile' + lane')) :
    r = s ∧ tile = tile' ∧ lane = lane' := by
  obtain ⟨rows, cols⟩ := splitAddress (Nat.lt_of_lt_of_le valid fits)
    (Nat.lt_of_lt_of_le valid' fits) same
  exact ⟨rows, splitAddress hl hl' cols⟩

-- Covers every lane, including masked lanes, in the 3072 × 12288 BF16
-- benchmark with 1024 columns per tile. The column is computed in signed
-- i32, the row offset in signed i64, and BF16 has two bytes per element.
example (row tile lane : Nat) (hr : row < 3072) (ht : tile < 12)
    (hl : lane < 1024) :
    1024 * tile + lane < 2^31 ∧
    2 * (12288 * row + (1024 * tile + lane)) < 2^63 := by omega

-- With row stride 2 at width 3, two distinct valid coordinates overlap.
example : 2 * 0 + 2 = 2 * 1 + 0 := by decide

end LigerAddresses
