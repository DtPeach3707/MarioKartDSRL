# Mario Kart DS Reinforcement Learning Program

This project was to make a DRL agent that learned to play the Figure-8 Circuit Course of Mario Kart DS (chosen partly because of childhood nostalgia and because of its not-too-sharp-turns and bridges that would have it utilize both screens)

It uses a DDQN Network, gets image inputs by taking a screenshot of the approximate area of the DS emualtor (fullscreen), and is rewarded based on a Lua Script that reads certain RAM adresses (speed, angle, and checkpoint) and determines speed and direction from them

It performs fairly well, being able to get to the final lap around 25% of the time by the 800th episode (out of 2000)

training still needs to be fully completed, which is something I plan to finish before 2022

More details are provided in the documentation pdf and code files
