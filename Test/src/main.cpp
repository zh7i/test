#include <iostream>
#include <string>
#include <ctime>
#include <cstdlib>

typedef double operand_type;


int main(int argc, char **argv) {
    int a = 10;
    float b = 0.1f;

    int c = *((int*)&b);
    float d = (float)c;


    return 0;
}
