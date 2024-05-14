#define CL_TARGET_OPENCL_VERSION 300
#define CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY

#include <CL/opencl.hpp>
#include <fstream>
#include <iostream>
#include <array>
#include <numeric>
#include <random>

void printIfError(cl_int errorCode) {
	if (errorCode != CL_SUCCESS) {
		std::cout << "Error code present! - " << errorCode << std::endl;
	}
}

cl::Program CreateProgram(const std::string& file)
{
	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);

	auto platform = platforms.front();
	std::vector<cl::Device> devices;
	platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);

	auto& device = devices.front();

	std::ifstream helloWorldFile(file);
	std::string src(std::istreambuf_iterator<char>(helloWorldFile), (std::istreambuf_iterator<char>()));

	cl::Program::Sources sources(1, std::make_pair(src.c_str(), src.length() + 1));

	cl::Context context(device);
	cl::Program program(context, sources);

	auto err = program.build("-cl-std=CL1.2");
	if (err != 0) {
		std::cout << "Program build error: " << err << std::endl;
	}
	return program;
}


std::vector<float> generateInputData(size_t size, float minValue, float maxValue) {
	std::vector<float> data(size);
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> dis(minValue, maxValue);
	for (size_t i = 0; i < size; ++i) {
		data.at(i) = dis(gen);
	}
	return data;
}

std::vector<float> generateFixedInputData(size_t size) {
	std::vector<float> data(size);
	for (size_t i = 0; i < size; ++i) {
		if (i % 2 == 0) {
			data.at(i) = static_cast<float>(1.5f);
		}
		else {
			data.at(i) = static_cast<float>(3.4f);
		}
	}
	return data;
}

std::vector<float> softmax(cl::Program program, cl::Context context, cl::Device& device, const int INPUT_SIZE, std::vector<float>& inputData) {
	// ---- Reduce Max ----
	cl::Kernel reduceMax(program, "reducemax");

	cl_int getWorkGroupErrorCode = 0;
	auto workGroupSize = reduceMax.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device, &getWorkGroupErrorCode);
	printIfError(getWorkGroupErrorCode);

	auto numWorkGroups = INPUT_SIZE / workGroupSize;

	cl::Buffer buf(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * INPUT_SIZE, inputData.data());
	cl::Buffer outbuf(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, sizeof(float) * INPUT_SIZE);

	reduceMax.setArg(0, buf);
	reduceMax.setArg(1, sizeof(float) * workGroupSize, nullptr);
	reduceMax.setArg(2, outbuf);

	std::vector<float> outVec(numWorkGroups);

	cl::CommandQueue queue(context, device);
	auto err2 = queue.enqueueNDRangeKernel(reduceMax, cl::NullRange, cl::NDRange(INPUT_SIZE), cl::NDRange(workGroupSize));
	printIfError(err2);
	auto err3 = queue.enqueueReadBuffer(outbuf, CL_TRUE, 0, sizeof(float) * INPUT_SIZE, outVec.data());
	printIfError(err3);
	queue.finish();

	float max = 0.f;
	std::vector<float> outCopy(outVec);
	if (outCopy.size() > 1) {
		// Should launch the max kernel again instead...
		max = *(std::max_element(outCopy.begin(), outCopy.end()));
	}

	std::cout << "Done Reduce Max" << std::endl;

	// ---- Subtract Max ----

	cl::Kernel subtractMaxKernel(program, "subtractmax");
	cl::Buffer subtractMaxInputBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * INPUT_SIZE, inputData.data());
	cl::Buffer subtractMaxOutputBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * INPUT_SIZE);
	subtractMaxKernel.setArg(0, subtractMaxInputBuffer);
	subtractMaxKernel.setArg(1, subtractMaxOutputBuffer);
	subtractMaxKernel.setArg(2, max);

	std::vector<float> outputSubtractedMax(INPUT_SIZE);
	auto err4 = queue.enqueueNDRangeKernel(subtractMaxKernel, cl::NullRange, cl::NDRange(INPUT_SIZE), cl::NDRange(workGroupSize));
	printIfError(err4);
	auto err5 = queue.enqueueReadBuffer(subtractMaxOutputBuffer, CL_TRUE, 0, sizeof(float) * INPUT_SIZE, outputSubtractedMax.data());
	printIfError(err5);
	queue.finish();

	std::cout << "Done Subtract Max" << std::endl;

	// ---- Reduce Sum ----

	cl::Kernel reduceSumKernel(program, "reducesum");
	cl::Buffer reduceSumInputBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * INPUT_SIZE, outputSubtractedMax.data());
	cl::Buffer reduceSumOutputBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, sizeof(float) * INPUT_SIZE);

	reduceSumKernel.setArg(0, reduceSumInputBuffer);
	reduceSumKernel.setArg(1, sizeof(float) * workGroupSize, nullptr);
	reduceSumKernel.setArg(2, reduceSumOutputBuffer);

	std::vector<float> reduceSumOut(numWorkGroups);

	auto err6 = queue.enqueueNDRangeKernel(reduceSumKernel, cl::NullRange, cl::NDRange(INPUT_SIZE), cl::NDRange(workGroupSize));
	printIfError(err2);
	auto err7 = queue.enqueueReadBuffer(reduceSumOutputBuffer, CL_TRUE, 0, sizeof(float) * INPUT_SIZE, reduceSumOut.data());
	printIfError(err3);
	queue.finish();

	float sum = 0.f;
	std::vector<float> outSumCopy(reduceSumOut);
	if (outSumCopy.size() > 1) {
		// Should launch the sum kernel again instead...
		sum = std::accumulate(outSumCopy.begin(), outSumCopy.end(), 0);
	}

	std::cout << "Done Reduce Sum" << std::endl;

	// ---- Divide Exponentials by Sum ----

	cl::Kernel divideExpBySumKernel(program, "divideexpbysum");
	cl::Buffer divideExpBySumInputBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * INPUT_SIZE, outputSubtractedMax.data());
	cl::Buffer divideExpBySumOutputBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * INPUT_SIZE);
	divideExpBySumKernel.setArg(0, divideExpBySumInputBuffer);
	divideExpBySumKernel.setArg(1, divideExpBySumOutputBuffer);
	divideExpBySumKernel.setArg(2, sum);

	std::vector<float> outputDivideExpBySum(INPUT_SIZE);
	auto err8 = queue.enqueueNDRangeKernel(divideExpBySumKernel, cl::NullRange, cl::NDRange(INPUT_SIZE), cl::NDRange(workGroupSize));
	printIfError(err4);
	auto err9 = queue.enqueueReadBuffer(divideExpBySumOutputBuffer, CL_TRUE, 0, sizeof(float) * INPUT_SIZE, outputDivideExpBySum.data());
	printIfError(err5);
	queue.finish();

	std::cout << "Done Divide Exponentials by Sum" << std::endl;

	// verify validity of softmax
	std::vector<float> resultSoftmaxCopy(outputDivideExpBySum);
	auto resultSoftmax = std::accumulate(resultSoftmaxCopy.begin(), resultSoftmaxCopy.end(), 1);
	std::cout << resultSoftmax << std::endl;
	
	return resultSoftmaxCopy;
}

int main()
{
	auto program = CreateProgram("kernels.cl");
	auto context = program.getInfo<CL_PROGRAM_CONTEXT>();
	auto devices = context.getInfo<CL_CONTEXT_DEVICES>();
	auto& device = devices.front();

	const int INPUT_SIZE = 512;

	// Generate random or fixed input data
	std::vector<float> inputData = generateFixedInputData(INPUT_SIZE); // Alternative: std::vector<float> inputData = generateInputData(INPUT_SIZE, -5.0f, 5.0f);
	std::vector<float> outputData = softmax(program, context, device, INPUT_SIZE, inputData);


	

	//std::cin.get();

}