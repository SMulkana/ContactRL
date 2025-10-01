# ContactRL
ContactRL provides research-ready reinforcement learning environments and baselines for contact-aware robot control. It models a UR3e arm in PyBullet, exposes safety costs from contact forces, and includes Lagrangian and CPO formulations alongside SB3 SAC training, progress bars, and Excel/TensorBoard logging.

**Intallation**

**1. Clone this repository**

`git clone https://github.com/SMulkana/ContactRL.git`

`cd ContactRL`

**2. Create a virtual environment (recommended)**

`python3 -m venv venv`

`source venv/bin/activate`   # On Linux/Mac

`venv\Scripts\activate`      # On Windows`

**3. Install dependencies**

python → 3.8.19

gymnasium → 0.28.1

pybullet → 3.2.6

stable-baselines3 → 2.4.0a8

numpy → 1.23.5

pandas → 2.0.3

matplotlib → 3.7.5

opencv-python (cv2) → 4.10.0.84

tqdm → 4.66.4

torch → 2.0.1+cu117 (CUDA 11.7 build)

Pillow (PIL) → 10.4.0

openpyxl → 3.1.5

**4. Install robot and gripper models**

- UR3e URDF & meshes:`https://github.com/UniversalRobots/Universal_Robots_ROS2_Description`

- Robotiq 2F-85 URDF:`https://github.com/a-price/robotiq_arg85_description`

- Alternative Robotiq URDF repo: `https://dei-gitlab.dei.unibo.it/lar/robotiq_2f_gripper_visualization`

Place these files inside a ROS package or in a directory that PyBullet can access.
Update paths in ur3e_env.py if needed.

**Usage**

**1. Run Training**

`train.py`

**2. Logs and results**

TensorBoard logs and trained models are saved.

During training, the environment logs not only the standard Stable-Baselines3 metrics (which can be monitored in TensorBoard) but also a set of custom parameters tailored to safe contact learning. These include step-wise Rewards, human robot interaction force (the normal force applied on the hand by the robot), Contact Type (e.g., safe_contact or no contact), as well as aggregated values such as Episodic Reward, Average Episode Reward, Episode end Condition, and Total time per episode. At the end of training, these values are saved into structured Excel files (.xlsx) for easy post-analysis. You can open these directly in Excel or load them into Python for plotting. This makes it straightforward to visualize how rewards evolve with contact forces, compare safety across runs, and analyze robot behavior without re-running simulations.

**3. Monitor training with TensorBoard**

You can monitor training progress with TensorBoard:

`tensorboard --logdir=/path/to/ContactRL`

By default, TensorBoard starts a local server at: `http://localhost:6006`

Open that link in your browser to view training curves, reward progress, and other diagnostics in real time.

**4. Evaluate**

Evaluate trained models with:

`eval.py`


