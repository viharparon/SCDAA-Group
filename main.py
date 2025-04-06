from SCUtils import main_q1, main_soft_lqr, run_critic_algorithm, run_actor_algorithm, run_actor_critic_algorithm
import matplotlib.pyplot as plt

# Q1 code
try:
    main_q1()
    plt.show()
except Exception as e:
    print("Came across an unexpected error:", e)

# Q2 code
try:
    main_soft_lqr()
    plt.show()
except Exception as e:
    print("Came across an unexpected error:", e)

# Q3 code
try:
    run_critic_algorithm()
    plt.show()
except Exception as e:
    print("Came across an unexpected error:", e)

# Q4 code
try:
    run_actor_algorithm()
    plt.show()
except Exception as e:
    print("Came across an unexpected error:", e)

# Q5 code - takes a while to run
try:
    run_actor_critic_algorithm()
    plt.show()
except Exception as e:
    print("Came across an unexpected error:", e)