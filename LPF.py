import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.ticker import MultipleLocator
import customtkinter as ctk
from tkinter import messagebox
from collections import defaultdict
from matplotlib import patheffects


class LPInputWindow:
    def __init__(self):
        self.root = ctk.CTk()
        self.root.title("Linear Programming Solver")
        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")

        self.problem_type = None
        self.a_obj = None
        self.b_obj = None
        self.constraints = []

        self.create_widgets()

    def create_widgets(self):
        self.type_frame = ctk.CTkFrame(self.root)
        self.type_frame.pack(padx=10, pady=10, fill="x")

        ctk.CTkLabel(self.type_frame, text="Problem Type:").pack(anchor="w")
        self.type_var = ctk.StringVar(value="max")
        ctk.CTkRadioButton(self.type_frame, text="Maximize", variable=self.type_var, value="max").pack(anchor="w")
        ctk.CTkRadioButton(self.type_frame, text="Minimize", variable=self.type_var, value="min").pack(anchor="w")

        self.obj_frame = ctk.CTkFrame(self.root)
        self.obj_frame.pack(padx=10, pady=10, fill="x")

        ctk.CTkLabel(self.obj_frame, text="Objective Function (Z = ax + by):").pack(anchor="w")

        self.a_entry = ctk.CTkEntry(self.obj_frame, placeholder_text="Coefficient for x (a)")
        self.a_entry.pack(padx=5, pady=5, fill="x")

        self.b_entry = ctk.CTkEntry(self.obj_frame, placeholder_text="Coefficient for y (b)")
        self.b_entry.pack(padx=5, pady=5, fill="x")

        self.constraints_frame = ctk.CTkFrame(self.root)
        self.constraints_frame.pack(padx=10, pady=10, fill="both", expand=True)

        ctk.CTkLabel(self.constraints_frame, text="Constraints:").pack(anchor="w")

        self.constraints_listbox = ctk.CTkTextbox(self.constraints_frame, height=100)
        self.constraints_listbox.pack(padx=5, pady=5, fill="both", expand=True)

        self.add_constraint_frame = ctk.CTkFrame(self.root)
        self.add_constraint_frame.pack(padx=10, pady=10, fill="x")

        ctk.CTkLabel(self.add_constraint_frame, text="Add Constraint (ax + by ≤/≥/= c):").pack(anchor="w")

        self.constraint_a_entry = ctk.CTkEntry(self.add_constraint_frame, placeholder_text="a")
        self.constraint_a_entry.pack(padx=5, pady=2, fill="x")

        self.constraint_b_entry = ctk.CTkEntry(self.add_constraint_frame, placeholder_text="b")
        self.constraint_b_entry.pack(padx=5, pady=2, fill="x")

        self.constraint_type_var = ctk.StringVar(value="<=")
        ctk.CTkRadioButton(self.add_constraint_frame, text="≤", variable=self.constraint_type_var, value="<=").pack(
            side="left", padx=5)
        ctk.CTkRadioButton(self.add_constraint_frame, text="≥", variable=self.constraint_type_var, value=">=").pack(
            side="left", padx=5)
        ctk.CTkRadioButton(self.add_constraint_frame, text="=", variable=self.constraint_type_var, value="=").pack(
            side="left", padx=5)

        self.constraint_c_entry = ctk.CTkEntry(self.add_constraint_frame, placeholder_text="c")
        self.constraint_c_entry.pack(padx=5, pady=5, fill="x")

        ctk.CTkButton(self.add_constraint_frame, text="Add Constraint", command=self.add_constraint).pack(pady=5)

        ctk.CTkButton(self.root, text="Solve and Visualize", command=self.solve).pack(padx=10, pady=10)

    def add_constraint(self):
        try:
            a = float(self.constraint_a_entry.get())
            b = float(self.constraint_b_entry.get())
            c = float(self.constraint_c_entry.get())
            typ = self.constraint_type_var.get()

            self.constraints.append((a, b, typ, c))
            self.constraints_listbox.insert("end", f"{a}x + {b}y {typ} {c}\n")

            self.constraint_a_entry.delete(0, "end")
            self.constraint_b_entry.delete(0, "end")
            self.constraint_c_entry.delete(0, "end")

        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers for the constraint coefficients")

    def solve(self):
        try:
            self.problem_type = self.type_var.get()
            self.a_obj = float(self.a_entry.get())
            self.b_obj = float(self.b_entry.get())

            if not self.constraints:
                messagebox.showerror("Error", "Please add at least one constraint")
                return

            self.root.destroy()
            visualize_problem(self.problem_type, self.a_obj, self.b_obj, self.constraints)

        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers for the objective function coefficients")


def find_intersection(a1, b1, c1, a2, b2, c2):
    A = np.array([[a1, b1], [a2, b2]])
    b = np.array([c1, c2])
    try:
        return tuple(np.linalg.solve(A, b))
    except np.linalg.LinAlgError:
        return None


def plot_constraint(ax, a, b, constraint_type, c, x_range, label):
    if b != 0:
        y = (c - a * x_range) / b
        line, = ax.plot(x_range, y, label=label, alpha=0.7)

        # Only show the constraint symbol on the line
        mid_idx = len(x_range) // 2
        x_mid = x_range[mid_idx]
        y_mid = y[mid_idx]

        if constraint_type == '<=':
            ax.text(x_mid, y_mid, ' ≤', color=line.get_color(), ha='left', va='center',
                    fontsize=10, path_effects=[patheffects.withStroke(linewidth=3, foreground="white")])
        elif constraint_type == '>=':
            ax.text(x_mid, y_mid, ' ≥', color=line.get_color(), ha='left', va='center',
                    fontsize=10, path_effects=[patheffects.withStroke(linewidth=3, foreground="white")])
        else:
            ax.text(x_mid, y_mid, ' =', color=line.get_color(), ha='left', va='center',
                    fontsize=10, path_effects=[patheffects.withStroke(linewidth=3, foreground="white")])
    else:
        x_val = c / a
        ax.axvline(x=x_val, label=label, alpha=0.7)
        y_mid = (ax.get_ylim()[1] + ax.get_ylim()[0]) * 0.5
        if constraint_type == '<=':
            ax.text(x_val, y_mid, '≤', color='black', ha='right', va='center',
                    fontsize=10, path_effects=[patheffects.withStroke(linewidth=3, foreground="white")])
        elif constraint_type == '>=':
            ax.text(x_val, y_mid, '≥', color='black', ha='left', va='center',
                    fontsize=10, path_effects=[patheffects.withStroke(linewidth=3, foreground="white")])
        else:  # '='
            ax.text(x_val, y_mid, '=', color='black', ha='center', va='bottom',
                    fontsize=10, path_effects=[patheffects.withStroke(linewidth=3, foreground="white")])


def is_feasible(x, y, constraints):
    if x < -1e-6 or y < -1e-6:
        return False

    for a, b, constraint_type, c in constraints:
        lhs = a * x + b * y
        if constraint_type == '<=' and lhs > c + 1e-6:
            return False
        elif constraint_type == '>=' and lhs < c - 1e-6:
            return False
        elif constraint_type == '=' and not np.isclose(lhs, c, atol=1e-6):
            return False
    return True


def plot_feasible_region(ax, constraints, x_range):
    y_range = np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 100)
    X, Y = np.meshgrid(x_range, y_range)

    feasible = np.ones_like(X, dtype=bool)

    feasible &= (X >= -1e-6)
    feasible &= (Y >= -1e-6)

    for a, b, constraint_type, c in constraints:
        lhs = a * X + b * Y
        if constraint_type == '<=':
            feasible &= (lhs <= c + 1e-6)
        elif constraint_type == '>=':
            feasible &= (lhs >= c - 1e-6)
        else:  # '='
            feasible &= np.isclose(lhs, c, atol=1e-6)

    ax.contourf(X, Y, feasible, levels=[0.5, 1.5], colors=['lime'], alpha=0.3)
    ax.contour(X, Y, feasible, levels=[0.5], colors=['darkgreen'], linewidths=2)


class ObjectiveFunctionDragger:
    def __init__(self, ax, a_obj, b_obj, problem_type, constraints, x_range):
        self.ax = ax
        self.a_obj = a_obj
        self.b_obj = b_obj
        self.problem_type = problem_type
        self.constraints = constraints
        self.x_range = x_range
        self.z = 0

        self.line, = ax.plot([], [], 'r--', linewidth=2, label='Objective Function')
        self.text = ax.text(0.05, 0.95, '', transform=ax.transAxes,
                            bbox=dict(facecolor='white', alpha=0.8))

        self.cid_press = ax.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.cid_motion = ax.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)

        ax_reset = plt.axes([0.8, 0.025, 0.1, 0.04])
        self.button_reset = Button(ax_reset, 'Reset')
        self.button_reset.on_clicked(self.reset)

        self.update_line()

    def update_line(self):
        if self.b_obj != 0:
            y = (self.z - self.a_obj * self.x_range) / self.b_obj
            self.line.set_data(self.x_range, y)
        else:
            x_val = self.z / self.a_obj
            self.line.set_data([x_val, x_val], self.ax.get_ylim())

        self.text.set_text(f'Z = {self.a_obj}x + {self.b_obj}y = {self.z:.2f}')
        self.ax.figure.canvas.draw()

    def on_press(self, event):
        if event.inaxes != self.ax or event.button != 1:
            return

        x, y = event.xdata, event.ydata
        self.z = self.a_obj * x + self.b_obj * y
        self.update_line()

    def on_motion(self, event):
        if event.inaxes != self.ax or event.button != 1:
            return

        x, y = event.xdata, event.ydata
        self.z = self.a_obj * x + self.b_obj * y
        self.update_line()

    def reset(self, event):
        self.z = 0
        self.update_line()


def visualize_problem(problem_type, a_obj, b_obj, constraints):
    x_intercepts = [c / a for a, b, typ, c in constraints if a != 0]
    x_min = min(x_intercepts + [-10]) if x_intercepts else -10
    x_max = max(x_intercepts + [10]) if x_intercepts else 10
    x_range = np.linspace(x_min * 1.2, x_max * 1.2, 100)

    fig, ax = plt.subplots(figsize=(12, 8))

    ax.axvline(0, color='blue', linestyle='-', linewidth=1, alpha=0.7, label='x ≥ 0')
    ax.axhline(0, color='green', linestyle='-', linewidth=1, alpha=0.7, label='y ≥ 0')

    ax.annotate('', xy=(0.9, 0.5), xytext=(0.1, 0.5),
                xycoords='axes fraction', arrowprops=dict(facecolor='blue', shrink=0.05))
    ax.annotate('', xy=(0.5, 0.9), xytext=(0.5, 0.1),
                xycoords='axes fraction', arrowprops=dict(facecolor='green', shrink=0.05))

    for i, (a, b, typ, c) in enumerate(constraints, 1):
        plot_constraint(ax, a, b, typ, c, x_range, f'Constraint {i}: {a}x + {b}y {typ} {c}')

    plot_feasible_region(ax, constraints, x_range)

    corners = []
    corner_constraint_counts = defaultdict(int)

    # Find all corner points and count how many constraints pass through each
    for a, b, typ, c in constraints:
        if a != 0:
            x_intercept = c / a
            if x_intercept >= -1e-6 and is_feasible(x_intercept, 0, constraints):
                point = (round(x_intercept, 6), 0)
                corners.append(point)
                corner_constraint_counts[point] += 1
        if b != 0:
            y_intercept = c / b
            if y_intercept >= -1e-6 and is_feasible(0, y_intercept, constraints):
                point = (0, round(y_intercept, 6))
                corners.append(point)
                corner_constraint_counts[point] += 1

    for i in range(len(constraints)):
        for j in range(i + 1, len(constraints)):
            a1, b1, typ1, c1 = constraints[i]
            a2, b2, typ2, c2 = constraints[j]
            intersection = find_intersection(a1, b1, c1, a2, b2, c2)
            if intersection and intersection[0] >= -1e-6 and intersection[1] >= -1e-6 and is_feasible(intersection[0],
                                                                                                      intersection[1],
                                                                                                      constraints):
                point = (round(intersection[0], 6), round(intersection[1], 6))
                corners.append(point)
                corner_constraint_counts[point] += 2  # Count both constraints

    if is_feasible(0, 0, constraints):
        point = (0, 0)
        corners.append(point)
        # Count how many constraints pass through (0,0)
        for a, b, typ, c in constraints:
            if np.isclose(a * 0 + b * 0, c, atol=1e-6):
                corner_constraint_counts[point] += 1

    # Remove duplicate corners and identify degenerate points
    unique_corners = list(set(corners))
    degenerate_points = [point for point in unique_corners if corner_constraint_counts[point] > 2]

    z_values = [a_obj * x + b_obj * y for (x, y) in unique_corners]

    if problem_type == 'max':
        optimal_value = max(z_values)
        optimal_indices = [i for i, z in enumerate(z_values) if np.isclose(z, optimal_value, atol=1e-6)]
        optimal_type = "Maximum"
    else:
        optimal_value = min(z_values)
        optimal_indices = [i for i, z in enumerate(z_values) if np.isclose(z, optimal_value, atol=1e-6)]
        optimal_type = "Minimum"

    optimal_points = [unique_corners[i] for i in optimal_indices]

    # Check for multiple optimal solutions (entire edge/face optimal)
    multiple_optima = False
    if len(optimal_points) > 1:
        multiple_optima = True

    # Mark all degenerate points with a different marker (details in legend)
    for point in degenerate_points:
        ax.plot(point[0], point[1], 's', markersize=10,
                markeredgecolor='purple', markerfacecolor='none', markeredgewidth=2,
                label=f'Degenerate point ({point[0]:.1f}, {point[1]:.1f})')

    # Mark all optimal points (details in legend)
    for i, point in enumerate(optimal_points):
        ax.plot(point[0], point[1], 'o', markersize=12,
                markeredgecolor='red', markerfacecolor='gold', markeredgewidth=2,
                label=f'Optimal {i + 1} ({point[0]:.1f}, {point[1]:.1f}) Z={optimal_value:.1f}')

    # If multiple optimal points, draw a line between them to show the optimal edge
    if multiple_optima and len(optimal_points) > 1:
        optimal_points_sorted = sorted(optimal_points, key=lambda p: p[0])  # Sort by x-coordinate
        x_vals = [p[0] for p in optimal_points_sorted]
        y_vals = [p[1] for p in optimal_points_sorted]
        ax.plot(x_vals, y_vals, 'r-', linewidth=3, alpha=0.5,
                label='Optimal edge (multiple solutions)')

    dragger = ObjectiveFunctionDragger(ax, a_obj, b_obj, problem_type, constraints, x_range)

    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title(f'Linear Programming Solution ({problem_type.title()} Z = {a_obj}x + {b_obj}y)',
                 fontsize=14, pad=20)

    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.grid(True, which='both', linestyle='--', alpha=0.7)

    # Create comprehensive legend outside the plot
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    max_abs_x = max(abs(xlim[0]), abs(xlim[1]))
    max_abs_y = max(abs(ylim[0]), abs(ylim[1]))
    ax.set_xlim(-max_abs_x, max_abs_x)
    ax.set_ylim(-max_abs_y, max_abs_y)
    fig.align_labels()
    plt.subplots_adjust(left=0.1, right=0.75, bottom=0.1, top=0.9)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(1))

    ax.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(MultipleLocator(0.5))

    ax.grid(which='both', alpha=0.2)

    # Prepare the message for the messagebox
    message = (f"{optimal_type} value of Z: {optimal_value:.2f}\n\n")

    if len(optimal_points) == 1:
        message += f"Unique optimal solution at x = {optimal_points[0][0]:.2f}, y = {optimal_points[0][1]:.2f}\n"
    else:
        message += "Multiple optimal solutions found at:\n"
        for i, point in enumerate(optimal_points, 1):
            message += f"Solution {i}: x = {point[0]:.2f}, y = {point[1]:.2f}\n"
        message += "\nAll points on the line connecting these solutions are optimal.\n"

    if degenerate_points:
        message += "\nDegeneracy detected at:\n"
        for point in degenerate_points:
            message += f"({point[0]:.2f}, {point[1]:.2f})\n"
        message += "\n(Degeneracy occurs when more than two constraints meet at a single vertex)"
    else:
        message += "\nNo degeneracy detected in the solution."

    messagebox.showinfo("Optimal Solution", message)
    plt.show()


def main():
    app = LPInputWindow()
    app.root.mainloop()


if __name__ == "__main__":
    main()