/*
Fast Artificial Neural Network Library (fann)
Copyright (C) 2003-2016 Steffen Nissen (steffen.fann@gmail.com)

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*/

#include "fann.h"


int main(int argc, char **argv)
{
    const unsigned int num_input  = 2;  // state and action
    const unsigned int num_output = 1;  // state-action 'Q' value
    const unsigned int num_layers = 3;
    const unsigned int num_neurons_hidden = 8;
    const float desired_error = (const float) 0.001;
    const unsigned int max_epochs = 50000;
    const unsigned int epochs_between_reports = 1000;

    struct fann *ann = fann_create_standard(num_layers, num_input, num_neurons_hidden, num_output);

    fann_set_activation_function_hidden(ann, FANN_SIGMOID);//_SYMMETRIC);
    fann_set_activation_function_output(ann, FANN_SIGMOID);//_SYMMETRIC);

    /*
        fann_set_training_algorithm()
        The default training algorithm is FANN_TRAIN_RPROP. (fann_train.h)
    */

    //fann_set_training_algorithm(ann, FANN_TRAIN_BATCH); // ver algs em fann_data.h

    fann_train_on_file(ann, "q-learn-data.txt", max_epochs, epochs_between_reports, desired_error);
    fann_save(ann, "q-learn.net");
    fann_print_connections(ann);
    fann_destroy(ann);

    return 0;
}
