# SCDAA-Group
SCDAA Group with Junming (s2649155), Mantas (1968924), and Melker (s2751496)

Contribution weights all equal i.e.: 1/3, 1/3 , 1/3

## How to run code:

*To run the code, clone the repo and execute main.py.*

The code is set up so that after each question is computed, plots will be displayed with the results. Those will need to be closed before the next question runs. To avoid that, you can remove all _plt.show()_ calls inside of main.py besides the final one to display all of the results at the very end. Some additional information is also printed out as the code runs if interested.

*Note*: The code for question 1 is scaled down to run for fewer time steps and sample counts since it otherwise takes a long time. We have included the plots of the results for the full required range in the report. 

The code is set up to run with the following parameters due to how long it takes to execute. You can modify these if you go into the final function of each question's module:
- Q1: time_steps = 1500, sample_counts = [2 * 4**i for i in range(5)] & num_samples = 1500,  time_steps_list = [2\**i for i in range(1, 10)]
- Q3: Time grid size 100, hidden_size=512, learning_rate=1e-3, num_epochs=500, batch_size=64, eval_interval=20
- Q4: num_epochs=500, batch_size=256, eval_interval=25, hidden_size=256, learning_rate=1e-3
- Q5: num_epochs=500, batch_size=16, eval_interval=25 actor_hidden_size=256, critic_hidden_size=512, actor_lr=1e-4, critic_lr=1e-3