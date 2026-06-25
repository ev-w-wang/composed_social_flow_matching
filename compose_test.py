import torch
import matplotlib.pyplot as plt
from line_fm.network import FM
from obstacle_conditioning.network import ObstacleConditioning
from line_fm.config import Config as line_fm_Config
from obstacle_conditioning.config import Config as obstacle_Config
from model_wrapper import ComposedModelWrapper, ComposedModelWrapperWithRepulsionField
from flow_matching.solver import ODESolver
from obstacle_conditioning.occupancy_utils import generate_occupancy, AnalyticRepulsionField
from obstacle_conditioning.occupancy_utils import AnalyticRepulsionField

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_points = line_fm_Config["num_points"]
state_dim = line_fm_Config["state_dim"]
grid_size = obstacle_Config["grid_size"]
n_samples = line_fm_Config["n_samples"]
alpha_start = .5
alpha_stop = 1.5
alpha_step = .1


def load_models():
    line_fm_model = FM(
        input_dim=line_fm_Config["input_dim"],
        num_points=num_points,
        state_dim=state_dim,
        hidden_dim=line_fm_Config["hidden_dim"],
    )
    line_fm_model.load_state_dict(torch.load("line_fm/model_final.pth", map_location=device))
    line_fm_model.to(device).eval()

    obstacle_model = ObstacleConditioning(
        grid_size=grid_size,
        num_points=num_points,
        state_dim=state_dim,
        hidden_dim=obstacle_Config["hidden_dim"],
        base_dim=obstacle_Config["base_dim"],
    )
    obstacle_model.load_state_dict(torch.load("obstacle_conditioning/model_final.pth", map_location=device))
    obstacle_model.to(device).eval()
    return line_fm_model, obstacle_model


def is_free(point, occupancy):
    x = int(point[0].clamp(0, grid_size - 1).item())
    y = int(point[1].clamp(0, grid_size - 1).item())
    return occupancy[0, 0, y, x] < 0.5


def sample_start_end(occupancy, min_dist=10.0, max_tries=200):
    for _ in range(max_tries):
        free = (occupancy[0, 0] < 0.5).nonzero(as_tuple=False)
        idx = free[torch.randint(free.shape[0], (2,))]
        points = torch.stack([idx[:, 1], idx[:, 0]], dim=1).float()
        start, end = points[0:1], points[1:2]
        if torch.norm(start - end) >= min_dist and is_free(start[0], occupancy) and is_free(end[0], occupancy):
            return start.to(device), end.to(device)
    raise RuntimeError("Could not sample start/end in free space")


def evaluate_trajectories(trajectories, start, end, occupancy, segment_steps=10):
    occ = occupancy[0, 0]
    n = trajectories.shape[0]

    alphas = torch.linspace(0, 1, num_points, device=trajectories.device).view(1, num_points, 1)
    expected = start.unsqueeze(1) + alphas * (end - start).unsqueeze(1)

    start_err = torch.norm(trajectories[:, 0] - start, dim=-1)
    end_err = torch.norm(trajectories[:, -1] - end, dim=-1)
    line_err = torch.norm(trajectories - expected, dim=-1).mean(dim=1)

    xs = trajectories[..., 0].long().clamp(0, grid_size - 1)
    ys = trajectories[..., 1].long().clamp(0, grid_size - 1)
    waypoint_hits = occ[ys, xs] > 0.5

    segment_hits = []
    for i in range(n):
        hits = 0
        for j in range(num_points - 1):
            for s in range(1, segment_steps):
                t = s / segment_steps
                pt = (1 - t) * trajectories[i, j] + t * trajectories[i, j + 1]
                x = int(pt[0].clamp(0, grid_size - 1).item())
                y = int(pt[1].clamp(0, grid_size - 1).item())
                if occ[y, x] > 0.5:
                    hits += 1
        segment_hits.append(hits)
    segment_hits = torch.tensor(segment_hits, dtype=torch.float32, device=trajectories.device)
    any_collision = waypoint_hits.any(dim=1) | (segment_hits > 0)

    return {
        "start_err_mean": start_err.mean().item(),
        "start_err_max": start_err.max().item(),
        "end_err_mean": end_err.mean().item(),
        "end_err_max": end_err.max().item(),
        "line_err_mean": line_err.mean().item(),
        "line_err_max": line_err.max().item(),
        "waypoint_collisions_mean": waypoint_hits.sum(dim=1).float().mean().item(),
        "segment_collisions_mean": segment_hits.mean().item(),
        "trajectories_with_collision": any_collision.sum().item(),
        "trajectories_total": n,
        "all_clear": any_collision.sum().item() == 0,
    }


def run_compose_test(line_fm_model, obstacle_model, occupancy, start, end, noise, alpha):
    input = torch.cat([start, end], dim=1)
    time_grid = torch.tensor([0.0, 1.0], device=device)
    solver = ODESolver(
        ComposedModelWrapper(line_fm_model, obstacle_model, input, noise, occupancy, alpha)
    )
    trajectories = solver.sample(
        x_init=noise,
        time_grid=time_grid,
        step_size=line_fm_Config["step_size"],
        method="midpoint",
        return_intermediates=False,
    )
    return trajectories, evaluate_trajectories(trajectories, start, end, occupancy)

def run_compose_test_with_repulsion_field(line_fm_model, repulsion_field, occupancy, start, end, noise, alpha):
    input = torch.cat([start, end], dim=1)
    time_grid = torch.tensor([0.0, 1.0], device=device)
    solver = ODESolver(
        ComposedModelWrapperWithRepulsionField(line_fm_model, repulsion_field, input, noise, alpha)
    )
    trajectories = solver.sample(
        x_init=noise,
        time_grid=time_grid,
        step_size=line_fm_Config["step_size"],
        method="midpoint",
        return_intermediates=False,
    )
    return trajectories, evaluate_trajectories(trajectories, start, end, occupancy)


def plot_trajectories(occupancy, trajectories, start, end, alpha, path):
    trajectories = trajectories.detach().cpu()
    occ_display = 1.0 - occupancy[0, 0].cpu().numpy()

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(occ_display, cmap="gray", origin="upper", vmin=0, vmax=1, extent=[0, grid_size, grid_size, 0])
    for i in range(trajectories.shape[0]):
        ax.plot(trajectories[i, :, 0], trajectories[i, :, 1], "o-", alpha=0.6, markersize=3)
    ax.plot(start[0, 0].cpu(), start[0, 1].cpu(), "go", markersize=10, label="start")
    ax.plot(end[0, 0].cpu(), end[0, 1].cpu(), "r*", markersize=12, label="end")
    ax.set_xlim(0, grid_size)
    ax.set_ylim(grid_size, 0)
    ax.set_title(f"Composed trajectories (alpha={alpha:.1f})")
    ax.set_xlabel("white = free, black = obstacle")
    ax.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close(fig)


def print_run_stats(alpha, stats):
    print(f"\n--- alpha={alpha:.1f} ---")
    print(f"  start error (mean/max):     {stats['start_err_mean']:.3f} / {stats['start_err_max']:.3f}")
    print(f"  end error (mean/max):         {stats['end_err_mean']:.3f} / {stats['end_err_max']:.3f}")
    print(f"  line error (mean/max):        {stats['line_err_mean']:.3f} / {stats['line_err_max']:.3f}")
    print(f"  waypoint collisions (mean):   {stats['waypoint_collisions_mean']:.2f}")
    print(f"  segment collisions (mean):    {stats['segment_collisions_mean']:.2f}")
    print(f"  trajectories with collision:  {stats['trajectories_with_collision']:.0f}/{stats['trajectories_total']}")
    print(f"  all trajectories clear:       {stats['all_clear']}")


def print_summary_table(results):
    print("\n=== Summary across alpha values ===")
    header = (
        f"{'alpha':>5} | {'start':>8} | {'end':>8} | {'line':>8} | "
        f"{'collisions':>10} | {'all clear':>9}"
    )
    print(header)
    print("-" * len(header))
    for alpha, stats in results:
        print(
            f"{alpha:5.1f} | "
            f"{stats['start_err_mean']:8.3f} | "
            f"{stats['end_err_mean']:8.3f} | "
            f"{stats['line_err_mean']:8.3f} | "
            f"{stats['trajectories_with_collision']:3.0f}/{stats['trajectories_total']:<6.0f} | "
            f"{str(stats['all_clear']):>9}"
        )


def main():
    line_fm_model, obstacle_model = load_models()
    occupancy = generate_occupancy(device, batch_size=1)
    rep_field = AnalyticRepulsionField(occupancy, device)   
    rep_field.plot_vector_field()
    # start, end = sample_start_end(occupancy)
    start = torch.tensor([[0.0, 0.0]], device=device)
    end = torch.tensor([[grid_size - 1, grid_size - 1]], device=device)
    noise = torch.randn(n_samples, num_points, state_dim, device=device)

    assert is_free(start[0], occupancy), "start must be in free space"
    assert is_free(end[0], occupancy), "end must be in free space"
    print(f"start: ({start[0, 0].item():.1f}, {start[0, 1].item():.1f})")
    print(f"end:   ({end[0, 0].item():.1f}, {end[0, 1].item():.1f})")

    alpha_values = torch.arange(alpha_start, alpha_stop + 1e-6, alpha_step).tolist()
    results = []
    best_alpha, best_stats, best_trajectories = None, None, None

    for alpha in alpha_values:
        # trajectories, stats = run_compose_test(
        #     line_fm_model, obstacle_model, occupancy, start, end, noise, alpha
        # )
        trajectories, stats = run_compose_test_with_repulsion_field(
            line_fm_model, rep_field, occupancy, start, end, noise, alpha
        )
        print_run_stats(alpha, stats)
        results.append((alpha, stats))
        plot_trajectories(occupancy, trajectories, start, end, alpha, f"test_images/compose_test_alpha_{alpha:.1f}.png")

        if best_stats is None or (
            stats["trajectories_with_collision"] < best_stats["trajectories_with_collision"]
            or (
                stats["trajectories_with_collision"] == best_stats["trajectories_with_collision"]
                and stats["end_err_mean"] < best_stats["end_err_mean"]
            )
        ):
            best_alpha, best_stats, best_trajectories = alpha, stats, trajectories

    print_summary_table(results)
    if best_trajectories is not None:
        plot_trajectories(occupancy, best_trajectories, start, end, best_alpha, "test_images/compose_test_best.png")
        print(f"\nBest alpha by fewest collisions / lowest end error: {best_alpha:.1f}")


if __name__ == "__main__":
    main()
