#include <iostream>
#include <string>

template <typename T1>struct TestFunctor1{
	T1 operator()(T1 input) {
		return input + input;
	}
};

template <typename T1, typename T2>struct TestStruct1{
	T1 operator()(T1 input, T2 functor) {
		return functor(input);
	}
};

// test 1
int main(int argc, char **argv) {

	TestFunctor1<int> functor;
	TestStruct1<int, struct TestFunctor1<int>> test;

	std::cout << test(1, functor) << std::endl;
	return 0;
}
