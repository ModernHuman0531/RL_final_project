# RL Final Project: Towards Trustworthy Smart Cities: An Explainable and Safe DRL Application for Sustainable Traffic Management

## Usage
* Build Docker image\
```zsh
make build
```
* Run image without GUI
```zsh
make run
```
* Run image with GUI
  1. For linux users(default)
   Check the `.env` files, please comment the code block for window users and uncomment the code block for linux users, then just run `make run-gui`
  2. For window users(WSL2)
     1. Install X server: Install VcXsrv
     2. Activate X Launch
        1. Choose "Multiple windows"
        2. Select "Disable access control"
     3. WSL2 command
        ```bash
        export DISPLAY=$(grep nameserver /etc/resolv.conf | awk '{print $2}'):0.0
        make run-gui
        ```

## File Structure
```txt
project/
├── nets/          # SUMO + scenario definition
│   ├── single_intersection/
│   │    ├── single_intersection.nod.xml
│   │    ├── single_intersection.edg.xml
│   │    ├── single_intersection.con.xml
│   │    ├── single_intersection.net.xml
│   │    └──  single_intersection.sumocfg
│   └── 2*2 grid/
│
├── env/              
│   └── sumo_env.py # Turn SUMO into RL problem(MDP)
│
├── agent/            # RL method (Learning policy + Constraint)
│   ├── base_agent.py
│   └── 
├── experiments/     # Training loop
│   ├── train.py
│   └── 
│
├── settings/
│   └── view.setting.xml # Display setting file      
│
├── models/
├── Makefile
├── Dockerfile
└── README.md
```
## Net design
### Baseline
The baseline is simple intersection, for simplicity, we:
* Constrain the car can only move straightly. (Use .con.xml to restrict this condition)
* Each line have two lanes for in and out.
* Speed limit is 50(km/h) (13.89(m/s)).
* Only allow the center node has traffic light.
* The project proporsal ask to be stationary poisson arrivals with a flow rate of around 300–600 vehicles/hour/lane. For simplicity, we fixed the flow rate to 360 vehicles/hour/lane. By using the probability parameter in `flow`, 360 vehicles devided by 3600 sec = 0.1 vechicle/sec, so set probability="0.1" means we have 10 percent of chance to generate a vehicle per second.
* For pedestrian, set 180 pedestrians/hour/lane, so probability="0.05"  
Use the following command to import the .sumocfg file into sumo-gui:
```zsh
sumo -c simple_intersection.sumocfg
```

For more advanced scenes, maybe we can first try to let the car can turn around or have more lanes or differ the car flow rate by time (like have the rush hour).
