from rlmep.exercise_2 import DiscreteMEP, GridSpec

def get_diffusion_state_grid(
    grid_size: tuple[int, int] = (20, 20),
    grid_spacing: float = 0.4,
    adsorbate_atom: str = "Cu",
    grid_shift: tuple[float, float] | None = None,
) -> DiscreteMEP:
    from ase.build import fcc100
    from ase.build import add_adsorbate

    # Define states:
    slab = fcc100("Cu", size=(5, 5, 2), vacuum=10.0)
    initial_state = slab.copy()
    add_adsorbate(
        initial_state, adsorbate_atom, height=1.5, position="hollow", offset=(1, 2)
    )
    final_state = slab.copy()
    add_adsorbate(
        final_state, adsorbate_atom, height=1.5, position="hollow", offset=(2, 2)
    )

    if grid_shift is None:
        grid_shift = (0.0, 0.0)

    index = len(initial_state) - 1
    dx = final_state.positions[index, 0] - initial_state.positions[index, 0]
    grid_spacing = dx / (dx // grid_spacing)

    # Grid
    c = initial_state.positions[-1]
    corner = (
        c[0] + grid_shift[0] * grid_spacing,
        c[1] + grid_shift[1] * grid_spacing,
        c[2],
    )

    gridspec = GridSpec(grid_size, grid_spacing, corner=corner, height=c[2])


    return initial_state, final_state, gridspec

def get_diffusion_env(
    grid_size: tuple[int, int] = (20, 20),
    grid_spacing: float = 0.4,
    max_steps: int = 100,
    barrier_max: float = 2.0,
    adsorbate_atom: str = "Cu",
    grid_shift: tuple[float, float] | None = None,
) -> DiscreteMEP:
    from ase.calculators.emt import EMT

    # Calculator:
    calculator = EMT()

    # Get initial and final states:
    initial_state, final_state, gridspec = get_diffusion_state_grid(
        grid_size=grid_size,
        grid_spacing=grid_spacing,
        adsorbate_atom=adsorbate_atom,
        grid_shift=grid_shift,
    )

    # Create the environment:

    env = DiscreteMEP(
        initial_config=initial_state,
        final_config=final_state,
        gridspec=gridspec,
        calculator=calculator,
        moving_atom=-1,  # The last atom is the adsorbate
        max_steps=max_steps,
        barrier_max=barrier_max)
    
    return env



