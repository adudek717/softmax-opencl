#define CL_TARGET_OPENCL_VERSION 300
#define CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY

#include <CL/opencl.hpp>
#include <fstream>
#include <iostream>
#include <array>
#include <numeric>

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

int main()
{
	auto programReduceMax = CreateProgram("reducemax.cl");
	//auto programSubtractMax = CreateProgram("subtractmax.cl");
	auto context = programReduceMax.getInfo<CL_PROGRAM_CONTEXT>();
	auto devices = context.getInfo<CL_CONTEXT_DEVICES>();
	auto& device = devices.front();

	// Prepare some example input
	const int numX = 8;
	const int numY = 8;
	const int numZ = 8;
	const int count = numX * numY * numZ;
	//std::array<std::array<std::array<float, numZ>, numY>, numX> inputArr;

	int num = 0;
	//for (int i = 0; i < numX; ++i) {
	//	for (int j = 0; j < numY; ++j) {
	//		for (int k = 0; k < numZ; ++k) {
	//			inputArr.at(i).at(j).at(k) = static_cast<float>(num++);
	//		}
	//	}
	//}
	std::vector<float> inputVec(count);
	for (int i = 0; i < count; ++i) {
		if (i % 2 == 0) {
			inputVec.at(i) = static_cast<float>(1.5f);
		}
		else {
			inputVec.at(i) = static_cast<float>(3.4f);
		}

	}

	// ---- Reduce Max ----

	cl::Kernel kernel(programReduceMax, "reducemax");

	cl_int err1 = 0;
	auto workGroupSize = kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device, &err1);
	printIfError(err1);

	auto numWorkGroups = count / workGroupSize;

	cl::Buffer buf(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * count, inputVec.data());
	cl::Buffer outbuf(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, sizeof(float) * count);

	kernel.setArg(0, buf);
	kernel.setArg(1, sizeof(float) * workGroupSize, nullptr);
	kernel.setArg(2, outbuf);

	std::vector<float> outVec(numWorkGroups);

	cl::CommandQueue queue(context, device);
	auto err2 = queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(count), cl::NDRange(workGroupSize));
	printIfError(err2);
	auto err3 = queue.enqueueReadBuffer(outbuf, CL_TRUE, 0, sizeof(float) * count, outVec.data());
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

	cl::Kernel subtractMaxKernel(programReduceMax, "subtractmax");
	cl::Buffer subtractMaxInputBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * count, inputVec.data());
	cl::Buffer subtractMaxOutputBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * count);
	subtractMaxKernel.setArg(0, subtractMaxInputBuffer);
	subtractMaxKernel.setArg(1, subtractMaxOutputBuffer);
	subtractMaxKernel.setArg(2, max);

	std::vector<float> outputSubtractedMax(count);
	auto err4 = queue.enqueueNDRangeKernel(subtractMaxKernel, cl::NullRange, cl::NDRange(count), cl::NDRange(workGroupSize));
	printIfError(err4);
	auto err5 = queue.enqueueReadBuffer(subtractMaxOutputBuffer, CL_TRUE, 0, sizeof(float) * count, outputSubtractedMax.data());
	printIfError(err5);
	queue.finish();

	std::cout << "Done Subtract Max" << std::endl;

	// ---- Reduce Sum ----

	cl::Kernel reduceSumKernel(programReduceMax, "reducesum");
	cl::Buffer reduceSumInputBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * count, outputSubtractedMax.data());
	cl::Buffer reduceSumOutputBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, sizeof(float) * count);

	reduceSumKernel.setArg(0, reduceSumInputBuffer);
	reduceSumKernel.setArg(1, sizeof(float) * workGroupSize, nullptr);
	reduceSumKernel.setArg(2, reduceSumOutputBuffer);

	std::vector<float> reduceSumOut(numWorkGroups);

	auto err6 = queue.enqueueNDRangeKernel(reduceSumKernel, cl::NullRange, cl::NDRange(count), cl::NDRange(workGroupSize));
	printIfError(err2);
	auto err7 = queue.enqueueReadBuffer(reduceSumOutputBuffer, CL_TRUE, 0, sizeof(float) * count, reduceSumOut.data());
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

	cl::Kernel divideExpBySumKernel(programReduceMax, "divideexpbysum");
	cl::Buffer divideExpBySumInputBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * count, outputSubtractedMax.data());
	cl::Buffer divideExpBySumOutputBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * count);
	divideExpBySumKernel.setArg(0, divideExpBySumInputBuffer);
	divideExpBySumKernel.setArg(1, divideExpBySumOutputBuffer);
	divideExpBySumKernel.setArg(2, sum);

	std::vector<float> outputDivideExpBySum(count);
	auto err8 = queue.enqueueNDRangeKernel(divideExpBySumKernel, cl::NullRange, cl::NDRange(count), cl::NDRange(workGroupSize));
	printIfError(err4);
	auto err9 = queue.enqueueReadBuffer(divideExpBySumOutputBuffer, CL_TRUE, 0, sizeof(float) * count, outputDivideExpBySum.data());
	printIfError(err5);
	queue.finish();

	std::cout << "Done Divide Exponentials by Sum" << std::endl;

	// verify validity of softmax
	std::vector<float> resultSoftmaxCopy(outputDivideExpBySum);
	auto resultSoftmax = std::accumulate(resultSoftmaxCopy.begin(), resultSoftmaxCopy.end(), 1);
	std::cout << resultSoftmax << std::endl;

	//std::cin.get();

}