#ifndef FANN_WRAPPER__H
#define FANN_WRAPPER__H

#include <string>
#include <iostream>
#include <vector>

#include "fann.h"

#define MAX_INPUTS 100

class FannInfer
{
public:

    FannInfer (std::string net, uint32_t nIn, uint32_t nOut): nInputs(nIn), nOutputs(nOut)
    {
        annModel = fann_create_from_file(net.c_str());
    }


    ~FannInfer()
    {
        fann_destroy(annModel);
    }


    void run(const std::vector<fann_type> & inputs)
    {
        if (inputs.size() != nInputs)
        {
            std::cout << "Error: trying to set invalid number of inputs\n";
            exit(0);
        }

        for (uint32_t i = 0; i < nInputs; i++)
            annInputs[i] = (fann_type)inputs[i];

            annOutputs = fann_run(annModel, annInputs);
    }

    fann_type getOutput (uint32_t i)
    {
        try
        {
            if (i >= nOutputs)
            throw i;
        }

        catch (uint32_t i)
        {
            std::cout << "Error: unspecified output\n";
            exit(0);
        }
        return annOutputs[i];
    }

    void printOutputs ()
    {
        for (uint32_t i = 0; i < nOutputs; i++)
            printf ("%f\n", annOutputs[i]);
        printf ("\n");
    }

    float setInputs (const std::vector<fann_type> & data)
    {
        if (data.size() != nInputs)
        {
            std::cout << "Error: trying to set invalid number of inputs\n";
            exit(0);
        }

        for (size_t i = 0; i < nInputs; i++)
            annInputs[i] = data[i];
    }

private:
    struct fann *annModel;
    uint32_t nInputs, nOutputs;
    fann_type *annOutputs;
    fann_type annInputs[MAX_INPUTS];

};

#endif