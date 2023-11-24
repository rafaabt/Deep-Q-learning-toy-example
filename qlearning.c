#include <stdio.h>
#include <stdlib.h>
#include <string.h>



/* Basic q-learning example taken from https://blog.floydhub.com/an-introduction-to-q-learning-reinforcement-learning/ 
   Converted into C code (the same parameters are used). */


//#define LOG_ANN_DATA  // Logs exploration data to train the ANN
#define USE_ANN       // Uses a simple ANN to compute the Q values for each action in a given state

#ifdef USE_ANN
    #include "fann.h"
#endif

#define MAX_STATES  9
#define MAX_ACTIONS 9
#define MAX_HOPS    9

#define gamma   0.75 // discount factor
#define alpha   0.9  // learning rate
#define epsilon 0.01 // prob. of taking random action (e-greedy tests)


typedef unsigned char     uint8_t;
typedef unsigned int      uint32_t;
typedef unsigned long int uint64_t;
typedef uint8_t           State_t;
typedef uint8_t           Action_t;

float Q[MAX_STATES][MAX_ACTIONS];

typedef enum 
{
    L1,
    L2,
    L3,
    L4,
    L5,
    L6,
    L7,
    L8,
    L9,
} eStates;



/*
    The robot environment is (https://blog.floydhub.com/an-introduction-to-q-learning-reinforcement-learning/)
     
    L1   L2  L3
    ---|
    L4 | L5 |L6
            ----
    L7   L8  L9

    Trained Q for final destination L1:

    L1: 43157.82    40421.92    0.00    0.00    0.00    0.00    0.00    0.00    0.00    
    L2: 40426.27    0.00    40113.20    0.00    40389.64    0.00    0.00    0.00    0.00    
    L3: 0.00    40185.93    0.00    0.00    0.00    39909.88    0.00    0.00    0.00    
    L4: 0.00    0.00    0.00    0.00    0.00    0.00    38430.05    0.00    0.00    
    L5: 0.00    40391.18    0.00    0.00    0.00    0.00    0.00    39050.58    0.00    
    L6: 0.00    0.00    40127.88    0.00    0.00    0.00    0.00    0.00    0.00    
    L7: 0.00    0.00    0.00    37988.06    0.00    0.00    0.00    39064.18    0.00    
    L8: 0.00    0.00    0.00    0.00    39293.68    0.00    38858.25    0.00    36361.58    
    L9: 0.00    0.00    0.00    0.00    0.00    0.00    0.00    38603.73    0.00    


    If the neural network is used:

        22136.83    21045.43    0.00    0.00    0.00    0.00    0.00    0.00    0.00    
        24937.28    0.00    21761.17    0.00    23753.92    0.00    0.00    0.00    0.00    
        0.00    19234.06    0.00    0.00    0.00    17629.57    0.00    0.00    0.00    
        0.00    0.00    0.00    0.00    0.00    0.00    16846.94    0.00    0.00    
        0.00    24217.82    0.00    0.00    0.00    0.00    0.00    18019.37    0.00    
        0.00    0.00    18928.83    0.00    0.00    0.00    0.00    0.00    0.00    
        0.00    0.00    0.00    16312.63    0.00    0.00    0.00    14428.46    0.00    
        0.00    0.00    0.00    0.00    19674.65    0.00    17781.95    0.00    14641.37    
        0.00    0.00    0.00    0.00    0.00    0.00    0.00    18165.06    0.00    
*/


const float rewards[MAX_STATES][MAX_ACTIONS] =
{
    {0,1,0,0,0,0,0,0,0},
    {1,0,1,0,1,0,0,0,0},
    {0,1,0,0,0,1,0,0,0},
    {0,0,0,0,0,0,1,0,0},
    {0,1,0,0,0,0,0,1,0},
    {0,0,1,0,0,0,0,0,0},
    {0,0,0,1,0,0,0,1,0},
    {0,0,0,0,1,0,1,0,1},
    {0,0,0,0,0,0,0,1,0}
};


float rewards_new[MAX_STATES][MAX_STATES];


int tryEvent (float prob)
{
	if (prob == 0)
		return 0;
	
	float probPercent = 100*prob;
	
	float p = rand()%101; // 0-100
	
	return p <= probPercent;
}

void print_Q ()
{
    for (State_t i = 0; i < MAX_STATES; i++)
    {
        for (Action_t j = 0; j < MAX_ACTIONS; j++)
            printf ("%.2f\t", Q[i][j]);
        printf ("\n");
    }
}


Action_t get_max_action (State_t s) // Returns the best action for the state s
{
    float max = -1;
    Action_t action;

    for (Action_t a = 0; a < MAX_ACTIONS; a++)
    {
        if (Q[s][a] > max)
        {
            max = Q[s][a];
            action = a;
        }
    }

    return action;
}


uint32_t get_optimal_route (State_t *route, State_t start_location, State_t end_location) // Exploit
{
    State_t next_location;

    route[0] = start_location;
    next_location = start_location;
    uint32_t k = 1;

    printf ("Best route from L%d to L%d: ", start_location+1, end_location+1);

    while (next_location != end_location && k < MAX_HOPS) 
    {
        next_location = get_max_action(start_location);
        route[k++] = next_location;
        start_location = next_location;

    }
    
    if (k == MAX_HOPS)
        printf ("[ANN err]: Path not found (bad ANN model?)\n");

    else
        for (uint32_t i = 0; i < k; i++)
            printf ("%d -> ", route[i]+1);
    
    printf ("\n");

    return k;
}

void train_q(State_t end_location) // Explore
{
    bzero (Q, sizeof(float)*MAX_STATES*MAX_ACTIONS);

    State_t ending_state = end_location;

    memcpy (rewards_new, rewards, sizeof(float)*MAX_STATES*MAX_STATES);
    rewards_new[ending_state][ending_state] = 999;

    Action_t playable_actions[MAX_ACTIONS];

    const uint32_t samples = 1000;

#ifdef LOG_ANN_DATA
    FILE *f = fopen ("q-learn-data.txt", "w");
    fprintf(f, "%d 2 1\n", samples);
#endif

#ifdef USE_ANN
  fann_type *ann_out;
  struct fann *ann = fann_create_from_file("q-learn.net");
  fann_type ann_input[2];
#endif

    for (uint32_t i = 0 ; i < samples; i++) // Traning phase
    {
        State_t current_state = rand()%MAX_STATES;
        uint32_t k = 0;
    
        bzero (playable_actions, sizeof(uint8_t)*MAX_ACTIONS);

        for (uint32_t j = 0; j < MAX_ACTIONS; j++)
            if (rewards_new[current_state][j] > 0)
                playable_actions[k++] = j; // takes all possible actions for current_state

     /* Explore: takes a random action, evaluates the result without previous knowledge ("off police") */

        Action_t rd_act = playable_actions[rand()%k]; 
        State_t next_state = rd_act;

    #ifdef USE_ANN  // If we want to use the ANN to estimate each of the Q[current_state][rd_act]
        ann_input[0] = (float)current_state/10.0;
        ann_input[1] = (float)rd_act/10.0; 
        ann_out = fann_run(ann, ann_input);
        Q[current_state][rd_act] =  ann_out[0]*100000.0;

    #else // No ANN used (the same case as the blog)
		
		Action_t max_act;
		
		if(tryEvent(epsilon))  // (E-greedy) takes a random action with probability epsilon
			max_act =  playable_actions[rand()%k];
		else
			max_act = get_max_action(next_state); // max_act is the best possible action in state next_state
		
        float td = rewards_new[current_state][rd_act] + gamma * (Q[next_state][max_act] - Q[current_state][rd_act]);
        Q[current_state][rd_act] += alpha*td;   // state-action value function
    #endif

    #ifdef LOG_ANN_DATA // Logs the data used to train the ANN
        fprintf(f, "%.1f %.1f\n%.10f\n", (float)current_state/10.0, (float)rd_act/10.0, Q[current_state][rd_act]/100000.0);
    #endif
    }

    #ifdef LOG_ANN_DATA
        fclose (f);
    #endif
}



int main (int argc, char **argv)
{
    State_t route[MAX_HOPS];
    bzero (route, sizeof(State_t)*MAX_HOPS);
    uint32_t totalLocs;

    /* Train to go to position L1 starting frorm any random position */
    train_q (L1);

    printf ("\nTrained Q to L1:\n");
    print_Q();
    printf ("\n");
 
    totalLocs = get_optimal_route(route, L2, L1);
    totalLocs = get_optimal_route(route, L3, L1);   
    totalLocs = get_optimal_route(route, L4, L1);
    totalLocs = get_optimal_route(route, L5, L1);
    totalLocs = get_optimal_route(route, L7, L1);

    totalLocs = get_optimal_route(route, L6, L1);  
    totalLocs = get_optimal_route(route, L8, L1);
    totalLocs = get_optimal_route(route, L9, L1);   

    totalLocs = get_optimal_route(route, L8, L1);
    totalLocs = get_optimal_route(route, L7, L1);
    totalLocs = get_optimal_route(route, L6, L1);
    totalLocs = get_optimal_route(route, L1, L1);

    return 0;
}

