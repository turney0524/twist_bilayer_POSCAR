#!/usr/bin/env python3
"""Build a commensurate twisted bilayer from a hexagonal 2D POSCAR.

All atomic species, coordinates, occupancies, and site properties in the input
monolayer are retained.  Thus the same script works for graphene, CuCrP2S6,
MoS2, etc., provided that the input is one isolated layer with a hexagonal
(triangular) in-plane primitive lattice.

Examples:
  # 21.79 degree twisted bilayer graphene
  python build_twisted_bilayer.py POSCAR -m 2 -n 1 -o POSCAR_tBG

  # CuCrP2S6, with a user-selected lateral translation of the top layer
  python build_twisted_bilayer.py POSCAR_CuCrP2S6 -m 3 -n 2 \
      --distance 6.8 --vacuum 28 --shift 0.333333 0.333333 -o POSCAR_tCuCrP2S6

The two values after --shift are fractional coordinates in the *input* a1/a2
basis.  They specify the translation of the top layer after its rotation.
"""

from __future__ import annotations

import argparse
from math import acos, degrees

import numpy as np
from pymatgen.core import Lattice, Structure
from pymatgen.io.vasp import Poscar

TOL = 2e-3


def rotation_about_axis(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Cartesian Rodrigues rotation matrix about a unit-vector ``axis``."""
    x, y, z = axis / np.linalg.norm(axis)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    cross = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return c * np.eye(3) + s * cross + (1.0 - c) * np.outer([x, y, z], [x, y, z])


def commensurate_angle(m: int, n: int) -> float:
    """Twist angle in radians for a hexagonal (m, n) coincidence cell."""
    cos_theta = (m * m + 4 * m * n + n * n) / (2 * (m * m + m * n + n * n))
    return acos(np.clip(cos_theta, -1.0, 1.0))


def as_sixty_degree_hexagonal(mono: Structure) -> Structure:
    """Return an equivalent primitive cell with a 60-degree a1/a2 angle.

    Hexagonal POSCARs are commonly written with either 60 or 120 degrees.  The
    coincidence-cell formula below uses 60 degrees, so the latter is rebased
    without changing the atomic structure or the cell area.
    """
    a1, a2, c = mono.lattice.matrix
    normal = np.cross(a1, a2)
    if np.linalg.norm(normal) < 1e-10:
        raise ValueError("The first two lattice vectors are collinear.")
    normal /= np.linalg.norm(normal)
    if np.linalg.norm(c - np.dot(c, normal) * normal) > TOL * np.linalg.norm(c):
        raise ValueError("The third lattice vector must be normal to the 2D layer plane.")

    gamma = degrees(np.arccos(np.clip(np.dot(a1, a2) / (np.linalg.norm(a1) * np.linalg.norm(a2)), -1, 1)))
    same_length = np.isclose(np.linalg.norm(a1), np.linalg.norm(a2), rtol=TOL, atol=TOL)
    if not same_length or not (np.isclose(gamma, 60.0, atol=0.15) or np.isclose(gamma, 120.0, atol=0.15)):
        raise ValueError(
            "This method needs a hexagonal/triangular in-plane lattice "
            f"(a=b, gamma=60 or 120 degrees); got a={np.linalg.norm(a1):.5f}, "
            f"b={np.linalg.norm(a2):.5f}, gamma={gamma:.4f} degrees."
        )
    if np.isclose(gamma, 60.0, atol=0.15):
        return mono

    # b1=a1, b2=a1+a2 changes gamma from 120 to 60 degrees.  det=1, so this
    # is only a change of lattice basis and does not enlarge the primitive cell.
    rebased = mono.copy()
    rebased.make_supercell([[1, 0, 0], [1, 1, 0], [0, 0, 1]])
    return rebased


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="one-layer hexagonal POSCAR")
    parser.add_argument("-m", type=int, default=2, help="commensurate integer m (default: 2)")
    parser.add_argument("-n", type=int, default=1, help="commensurate integer n (default: 1)")
    parser.add_argument("-o", "--output", default="POSCAR_twisted_bilayer", help="output POSCAR")
    parser.add_argument("--distance", type=float, default=3.35, help="distance between mean layer planes, in A")
    parser.add_argument("--vacuum", type=float, default=20.0, help="total out-of-plane cell height, in A")
    parser.add_argument("--shift", type=float, nargs=2, metavar=("U", "V"), default=(0.0, 0.0),
                        help="top-layer shift U*a1 + V*a2 after rotation (default: 0 0)")
    args = parser.parse_args()

    if args.m <= args.n or args.n <= 0:
        raise ValueError("Use integers m > n > 0 (for example m=2, n=1).")
    if args.distance <= 0:
        raise ValueError("--distance must be positive.")

    mono = as_sixty_degree_hexagonal(Structure.from_file(args.input))
    a1, a2 = mono.lattice.matrix[:2]
    axis = np.cross(a1, a2)
    axis /= np.linalg.norm(axis)

    # The (m,n) construction is universal for any atomic basis on a hexagonal
    # Bravais lattice--not only for a one-element graphene basis.
    mb = np.array([[args.m, args.n, 0], [-args.n, args.m + args.n, 0], [0, 0, 1]])
    mt = np.array([[args.n, args.m, 0], [-args.m, args.m + args.n, 0], [0, 0, 1]])
    bottom, top = mono.copy(), mono.copy()
    bottom.make_supercell(mb)
    top.make_supercell(mt)

    theta = commensurate_angle(args.m, args.n)
    # This sign maps the top coincidence lattice to the bottom lattice.
    top_cart = top.cart_coords @ rotation_about_axis(axis, -theta).T
    bottom_cart = bottom.cart_coords.copy()

    # Centre the actual layer, retaining internal buckling and all sublayers.
    mean_plane = np.mean(mono.cart_coords @ axis)
    layer_thickness = np.ptp(mono.cart_coords @ axis)
    if args.vacuum <= args.distance + layer_thickness:
        raise ValueError(
            f"--vacuum ({args.vacuum:g} A) must exceed distance + layer thickness "
            f"({args.distance + layer_thickness:.3f} A)."
        )
    bottom_cart += np.outer(np.ones(len(bottom)), axis * (args.vacuum / 2 - args.distance / 2 - mean_plane))
    top_cart += np.outer(np.ones(len(top)), axis * (args.vacuum / 2 + args.distance / 2 - mean_plane))
    top_cart += np.outer(np.ones(len(top)), args.shift[0] * a1 + args.shift[1] * a2)

    final_lattice = Lattice(np.vstack((bottom.lattice.matrix[:2], axis * args.vacuum)))
    species = [site.species for site in bottom] + [site.species for site in top]
    site_properties = {
        name: list(bottom.site_properties[name]) + list(top.site_properties[name])
        for name in set(bottom.site_properties) | set(top.site_properties)
    }
    bilayer = Structure(
        final_lattice, species, np.vstack((bottom_cart, top_cart)), coords_are_cartesian=True,
        to_unit_cell=True, site_properties=site_properties,
    )
    Poscar(bilayer, comment=(f"twisted bilayer: {mono.composition.reduced_formula}; "
                             f"m={args.m}, n={args.n}, theta={degrees(theta):.6f} deg; "
                             f"shift=({args.shift[0]:g},{args.shift[1]:g})")).write_file(args.output)

    area_factor = args.m * args.m + args.m * args.n + args.n * args.n
    print(f"input layer: {mono.composition.reduced_formula}, {len(mono)} atoms")
    print(f"twist angle: {degrees(theta):.6f} deg")
    print(f"moire area factor: {area_factor}")
    print(f"atoms: {len(bilayer)} ({len(mono)} x 2 x {area_factor})")
    print(f"written: {args.output}")


if __name__ == "__main__":
    main()
