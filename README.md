# ML Controls Simulator

The purpose of this repository is to create a system to validate [NNFF](https://github.com/nworb-cire/NNFF) and other control algorithms.

The code in this repository aims to solve the forward dynamics problem for a car, i.e. predict the future system state given the current state and inputs. The code takes some inspiration from Comma's [controls challenge](https://github.com/commaai/controls_challenge). It also seeks to provide a simple interface for testing control algorithms.

## Forward Dynamics

The forward dynamics problem is to predict the future lateral acceleration of a car given the input values of current and past lateral acceleration, road roll, steering angle, velocity, and acceleration. The trained model can then be used to simulate the car's behavior in a control loop, where the controller is given a target lateral acceleration and must predict the steering angle to achieve that target. The validation error of the controller is measured as the difference between the target and predicted actual lateral acceleration.
