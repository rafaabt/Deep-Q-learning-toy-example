#include <stdio.h>
#include <stdlib.h>
#include <string.h>


//#define LOG_ANN_DATA
#define USE_ANN           // Uses a simple ANN to compute the Q values for each action in a given state


#ifdef USE_ANN
    #include "fann.h"
#endif



/* Implementação do algoritmo em https://blog.floydhub.com/an-introduction-to-q-learning-reinforcement-learning/ 
    convertido de Python para C (os mesmos parâmetros são usados)
*/


#define MAX_STATES  9
#define MAX_ACTIONS 9
#define MAX_HOPS    9

#define gamma  0.75
#define alpha  0.9

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
    Exeplo de environment (https://blog.floydhub.com/an-introduction-to-q-learning-reinforcement-learning/)
     

    L1   L2  L3
    ---|
    L4 | L5 |L6
            ----
    L7   L8  L9


    Exemplo de saída, se treinar o robô para chegar até a posilão L1 a partir de qq ponto de saída aleatório (exploiting)

    Trained Q:

    L1: 43157.82    40421.92    0.00    0.00    0.00    0.00    0.00    0.00    0.00    
    L2: 40426.27    0.00    40113.20    0.00    40389.64    0.00    0.00    0.00    0.00    
    L3: 0.00    40185.93    0.00    0.00    0.00    39909.88    0.00    0.00    0.00    
    L4: 0.00    0.00    0.00    0.00    0.00    0.00    38430.05    0.00    0.00    
    L5: 0.00    40391.18    0.00    0.00    0.00    0.00    0.00    39050.58    0.00    
    L6: 0.00    0.00    40127.88    0.00    0.00    0.00    0.00    0.00    0.00    
    L7: 0.00    0.00    0.00    37988.06    0.00    0.00    0.00    39064.18    0.00    
    L8: 0.00    0.00    0.00    0.00    39293.68    0.00    38858.25    0.00    36361.58    
    L9: 0.00    0.00    0.00    0.00    0.00    0.00    0.00    38603.73    0.00    


    Saída usando rede neural:
        22136.83    21045.43    0.00    0.00    0.00    0.00    0.00    0.00    0.00    
        24937.28    0.00    21761.17    0.00    23753.92    0.00    0.00    0.00    0.00    
        0.00    19234.06    0.00    0.00    0.00    17629.57    0.00    0.00    0.00    
        0.00    0.00    0.00    0.00    0.00    0.00    16846.94    0.00    0.00    
        0.00    24217.82    0.00    0.00    0.00    0.00    0.00    18019.37    0.00    
        0.00    0.00    18928.83    0.00    0.00    0.00    0.00    0.00    0.00    
        0.00    0.00    0.00    16312.63    0.00    0.00    0.00    14428.46    0.00    
        0.00    0.00    0.00    0.00    19674.65    0.00    17781.95    0.00    14641.37    
        0.00    0.00    0.00    0.00    0.00    0.00    0.00    18165.06    0.00    



    Exemplo, se o robô partir do ponto L3 (função get_optimal_route):
        Pega ação de maior valor na Linha L3 (40185.93), que leva o robô até o estado L2
        Faz o mesmo la linha L2, isto é, pega o valor 40426.27, que leva o robô ao estado L1
        No estado L1 é determinado a condição de parad


      Rota[L2->L1]: 2   1   
      Rota[L3->L1]: 3   2   1   
      Rota[L4->L1]: 4   7   8   5   2   1   
      Rota[L5->L1]: 5   2   1   
      Rota[L6->L1]: 6   3   2   1   
      Rota[L7->L1]: 7   8   5   2   1   
      Rota[L8->L1]: 8   5   2   1   
      Rota[L9->L1]: 9   8   5   2   1
*/


const float rewards[MAX_STATES][MAX_ACTIONS] =  // Mapeia estados para ações. Ex.: se estiver no estado L6, a única "playable action" é a que leva ao estado 3
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


void print_Q ()
{
    for (State_t i = 0; i < MAX_STATES; i++)
    {
        for (Action_t j = 0; j < MAX_ACTIONS; j++)
            printf ("%.2f\t", Q[i][j]);
        printf ("\n");
    }
}


Action_t get_max_action (State_t s) // Pega ação de maior Q para o estado s
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

    while (next_location != end_location && k < MAX_HOPS) // Percorre tabela Q até chegar no destino
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

    for (uint32_t i = 0 ; i < samples; i++) // Treina, calcula valores de Q
    {
        State_t current_state = rand()%MAX_STATES;
        uint32_t k = 0;
    
        bzero (playable_actions, sizeof(uint8_t)*MAX_ACTIONS);

        for (uint32_t j = 0; j < MAX_ACTIONS; j++)
        {
            if (rewards_new[current_state][j] > 0)
            {
                playable_actions[k++] = j; // preenche ações possíveis para este estado (current_state)
            }
        }


     /* Explore: Pega uma ação possível random: treino baseado em ações aleatórias em que não se sabe o quão boa é a ação antes de tomar a ação ("off police") */

        Action_t rd_act = playable_actions[rand()%k]; 
        State_t next_state = rd_act; //  Pela forma que os rewards foram modelados, isto mapeia ação em estado. rd_act: ação que levou ao estado next_state.
      
        /*  
            Se usar uma ANN pra prever o reward (acho):
                Treina a ANN pra receber como entrada o estado e uma ação por vez. A saída é o valor de Q[CurrentState][action] (uma saída pra cada action)

                CurrentState s               -> x--x
                for (all possible action ai) -> x--x  -> out (Q[s][ai])

                A rede roda uma vez pra cada ação.
                Ver para qual caso de action a saída da rede é maximizada
                Tomar a ação (max_act <- (.))

                Pra treinar a rede na FANN, logar:
                    current_state rd_act      (duas entradas)
                    Q[current_state][rd_act]  (uma saída)

                Ver sections II.B e II.C: https://arxiv.org/pdf/1712.00162.pdf

                "The input to the neural network is a state s, and the outputs are approximated q values for 
                different actions Q = {q (s, a; θ) |a ∈ As }."
        */

        Action_t max_act = get_max_action(next_state); // max_act é a melhor ação possível no estado next_state

    #ifdef USE_ANN
        ann_input[0] = (float)current_state/10.0;
        ann_input[1] = (float)rd_act/10.0; 

        ann_out = fann_run(ann, ann_input);

        Q[current_state][rd_act] =  ann_out[0]*100000.0;

    #else 
        float td = rewards_new[current_state][rd_act] + gamma * (Q[next_state][max_act] - Q[current_state][rd_act]);
        Q[current_state][rd_act] += alpha*td;   // state-action value function
    #endif

    #ifdef LOG_ANN_DATA

        #if 1
            fprintf(f, "%.1f %.1f\n%.10f\n", (float)current_state/10.0, (float)rd_act/10.0, Q[current_state][rd_act]/100000.0);
        #else // TODO: Treina o valor de Q para todas as ações possíveis de current_state
            fprintf(f, "%.1f\n", (float)current_state/10.0);
            for (uint32_t ii = 0; ii < MAX_ACTIONS; ii++)   
                fprintf(f, "%.10f\n", Q[current_state][ii]/100000.0);
            
        #endif
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


    /* Treinar o robô para ir até a polisição L1, dado que o robô inicia em qq posição aleatória */
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

    train_q (L3);
    
    printf ("\nTrained Q to L3:\n");
    print_Q();
    printf ("\n");

    totalLocs = get_optimal_route(route, L9, L3);
    totalLocs = get_optimal_route(route, L8, L3);
    totalLocs = get_optimal_route(route, L7, L3);
    totalLocs = get_optimal_route(route, L6, L3);
    totalLocs = get_optimal_route(route, L3, L3);

    return 0;
}