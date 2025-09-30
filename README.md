# Project Overview

CakeBot is a multi‑robot baking cell designed to automate cake preparation in commercial kitchen environments. Four coordinated robots retrieve ingredients, measure and mix batter, and manage oven loading/unloading. The system improves consistency, hygiene, and throughput while reducing manual handling of hot or heavy items.

# Key Features

* **Dual‑source Ingredient Handling:** One robot retrieves chilled items (fridge), another retrieves dry goods (cupboards).
* **7 Degrees of Freedom (optional):** One arm can be mounted on a linear rail for extended reach across wide storage.
* **Optimised Waypoints & Poses:** Pre‑taught pick/place and pour poses ensure smooth, precise movements in a compact kitchen layout.
* **Separation of Tasks:** Robots specialise in retrieval, dosing/mixing, and oven handling for clearer workflows and easier fault isolation.
* **Safety & Efficiency:** Guarded cell with interlocks around hot zones, spill detection, and immediate stop via GUI and hardware E‑STOP.

# System Components

* **Robot 1 – Fridge Picker:** Responsible for collecting eggs, milk, and butter from cold storage.
* **Robot 2 – Cupboard Picker:** Retrieves flour, sugar, and other dry ingredients from shelves/cupboards.
* **Robot 3 – Mixer & Doser:** Places and locks the bowl on a scale, doses ingredients, and mixes to recipe timings.
* **Robot 4 – Oven Handler:** Loads filled tins into the oven, manages timers, and unloads when baking completes.
* **Linear Rail (optional):** Extends the operational reach of one arm (typically the oven or fridge robot) to cover wider storage spans.

# Applications

* **Commercial baking and catering:** Repeatable batches with consistent quality.
* **Ghost kitchens / central production units:** Scalable throughput for standardised SKUs.
* **Food R&D / test kitchens:** Controlled trials of mixing times, hydration, and bake profiles.

# Installation & Setup

## Hardware Setup:

* Mount and anchor each robot at its station (fridge, pantry, mixing, oven).
* Install and align the linear rail (if used) on the designated robot.
* Connect power, networking, and I/O (scales, oven door/ready signals, E‑STOPs).
* Fit food‑grade end‑effectors (grippers, scoops, ladles) and heat shields for the oven robot.
* Calibrate scales and teach critical waypoints (shelf slots, bowl, tin, oven rack).

## Software Installation:

* Clone the CakeBot repository.
* Install dependencies (robot control libraries, motion planning tools, HMI/GUI package).
* Import or create recipe files (ingredients, quantities, mix durations, bake time/temperature).
* Configure robot models, storage locations, and safety zones in the system settings.

## Running the System:

* Start the CakeBot control software and GUI.
* Select a recipe and batch size; verify inventory and pre‑heat settings.
* Begin operation; the GUI displays progress, timers, and any prompts for operator acknowledgement.

# Usage Instructions

1. **Define the recipe task** in the configuration (ingredients, target weights/volumes, mix profile, bake time).
2. **Load the task** into the control software and confirm stations are stocked and the bowl/tin are in place.
3. **Start the run.** Robots autonomously fetch, dose, mix, and manage baking with hand‑offs at defined stations.
4. **Monitor via GUI.** View scale readings, timers, and safety status; respond to any prompts (e.g., tray swap).
5. **Completion.** The system unloads the oven and stages the cake for cooling/serving; batch data is logged.

# Contact

For more information or inquiries, please contact the project team
