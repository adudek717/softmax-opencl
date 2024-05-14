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

cl::size_type getWorkgroupSize(cl::Device& device, cl::Kernel& kernel, const int INPUT_SIZE) {
	cl_int getWorkGroupErrorCode = 0;
	auto workGroupSize = kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device, &getWorkGroupErrorCode);
	printIfError(getWorkGroupErrorCode);
	return workGroupSize;
}

std::vector<float> reduceMax(cl::Program& program, cl::Context& context, cl::Device& device, cl::CommandQueue& queue, const int INPUT_SIZE, std::vector<float>& inputData) {
	cl::Kernel reduceMaxKernel(program, "reducemax");

	auto workGroupSize = getWorkgroupSize(device, reduceMaxKernel, INPUT_SIZE);
	auto numWorkGroups = INPUT_SIZE / workGroupSize;

	cl::Buffer inputBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * INPUT_SIZE, inputData.data());
	cl::Buffer outputBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, sizeof(float) * INPUT_SIZE);

	reduceMaxKernel.setArg(0, inputBuffer);
	reduceMaxKernel.setArg(1, sizeof(float) * workGroupSize, nullptr);
	reduceMaxKernel.setArg(2, outputBuffer);

	std::vector<float> outVec(numWorkGroups);

	auto enqeuueNDRangeKernelErrorCode = queue.enqueueNDRangeKernel(reduceMaxKernel, cl::NullRange, cl::NDRange(INPUT_SIZE), cl::NDRange(workGroupSize));
	printIfError(enqeuueNDRangeKernelErrorCode);
	auto enqueueReadBufferErrorCode = queue.enqueueReadBuffer(outputBuffer, CL_TRUE, 0, sizeof(float) * outVec.size(), outVec.data());
	printIfError(enqueueReadBufferErrorCode);
	queue.finish();

	std::vector<float> outVecCopy = outVec;
	return outVecCopy;
}

std::vector<float> subtractMaxAndExp(cl::Program& program, cl::Context& context, cl::Device& device, cl::CommandQueue& queue, const int INPUT_SIZE, std::vector<float>& inputData, float max) {
	cl::Kernel subtractMaxAndExpKernel(program, "subtractmax");

	auto workGroupSize = getWorkgroupSize(device, subtractMaxAndExpKernel, INPUT_SIZE);
	auto numWorkGroups = INPUT_SIZE / workGroupSize;

	cl::Buffer subtractMaxInputBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * INPUT_SIZE, inputData.data());
	cl::Buffer subtractMaxOutputBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * INPUT_SIZE);

	subtractMaxAndExpKernel.setArg(0, subtractMaxInputBuffer);
	subtractMaxAndExpKernel.setArg(1, subtractMaxOutputBuffer);
	subtractMaxAndExpKernel.setArg(2, max);

	std::vector<float> outputSubtractedMax(INPUT_SIZE);
	auto enqeuueNDRangeKernelErrorCode = queue.enqueueNDRangeKernel(subtractMaxAndExpKernel, cl::NullRange, cl::NDRange(INPUT_SIZE), cl::NDRange(workGroupSize));
	printIfError(enqeuueNDRangeKernelErrorCode);
	auto enqueueReadBufferErrorCode = queue.enqueueReadBuffer(subtractMaxOutputBuffer, CL_TRUE, 0, sizeof(float) * outputSubtractedMax.size(), outputSubtractedMax.data());
	printIfError(enqueueReadBufferErrorCode);
	queue.finish();

	return outputSubtractedMax;
}

std::vector<float> reduceSum(cl::Program& program, cl::Context& context, cl::Device& device, cl::CommandQueue& queue, const int INPUT_SIZE, std::vector<float>& inputData) {
	cl::Kernel reduceSumKernel(program, "reducesum");

	auto workGroupSize = getWorkgroupSize(device, reduceSumKernel, INPUT_SIZE);
	auto numWorkGroups = INPUT_SIZE / workGroupSize;

	cl::Buffer reduceSumInputBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * INPUT_SIZE, inputData.data());
	cl::Buffer reduceSumOutputBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, sizeof(float) * INPUT_SIZE);

	reduceSumKernel.setArg(0, reduceSumInputBuffer);
	reduceSumKernel.setArg(1, sizeof(float) * workGroupSize, nullptr);
	reduceSumKernel.setArg(2, reduceSumOutputBuffer);

	std::vector<float> reduceSumOut(numWorkGroups);

	auto enqeuueNDRangeKernelErrorCode = queue.enqueueNDRangeKernel(reduceSumKernel, cl::NullRange, cl::NDRange(INPUT_SIZE), cl::NDRange(workGroupSize));
	printIfError(enqeuueNDRangeKernelErrorCode);
	auto enqueueReadBufferErrorCode = queue.enqueueReadBuffer(reduceSumOutputBuffer, CL_TRUE, 0, sizeof(float) * reduceSumOut.size(), reduceSumOut.data());
	printIfError(enqueueReadBufferErrorCode);
	queue.finish();

	return reduceSumOut;
}

std::vector<float> divideExpBySum(cl::Program& program, cl::Context& context, cl::Device& device, cl::CommandQueue& queue, const int INPUT_SIZE, std::vector<float>& inputData, const float sum) {
	cl::Kernel divideExpBySumKernel(program, "divideexpbysum");

	auto workGroupSize = getWorkgroupSize(device, divideExpBySumKernel, INPUT_SIZE);
	auto numWorkGroups = INPUT_SIZE / workGroupSize;

	cl::Buffer divideExpBySumInputBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * INPUT_SIZE, inputData.data());
	cl::Buffer divideExpBySumOutputBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * INPUT_SIZE);
	divideExpBySumKernel.setArg(0, divideExpBySumInputBuffer);
	divideExpBySumKernel.setArg(1, divideExpBySumOutputBuffer);
	divideExpBySumKernel.setArg(2, sum);

	std::vector<float> outputDivideExpBySum(INPUT_SIZE);
	auto enqeuueNDRangeKernelErrorCode = queue.enqueueNDRangeKernel(divideExpBySumKernel, cl::NullRange, cl::NDRange(INPUT_SIZE), cl::NDRange(workGroupSize));
	printIfError(enqeuueNDRangeKernelErrorCode);
	auto enqueueReadBufferErrorCode = queue.enqueueReadBuffer(divideExpBySumOutputBuffer, CL_TRUE, 0, sizeof(float) * outputDivideExpBySum.size(), outputDivideExpBySum.data());
	printIfError(enqueueReadBufferErrorCode);
	queue.finish();

	return outputDivideExpBySum;
}

std::vector<float> softmax(cl::Program program, cl::Context context, cl::Device& device, cl::CommandQueue& queue, const int INPUT_SIZE, const std::vector<float>& inputData) {
	// ---- Step 1. Reduce Max ----
	std::vector<float> inputForReduceMax(inputData);
	std::vector<float> reduceMaxResult = reduceMax(program, context, device, queue, INPUT_SIZE, inputForReduceMax);

	float max = 0.f;
	if (reduceMaxResult.size() > 1) {
		auto maxIt = std::max_element(reduceMaxResult.begin(), reduceMaxResult.end());
		max = *maxIt;
		std::cout << "ReduceMax - max value: " << max << std::endl;
	}
	else if (reduceMaxResult.size() == 1)
	{
		max = reduceMaxResult.at(0);
		std::cout << "ReduceMax - max value: " << max << std::endl;
	}
	else
	{
		std::cout << "ERROR: Reduce Max return nothing!" << std::endl;
	}


	// ---- Step 2. Subtract Max And Exp ----
	std::vector<float> inputForSubtractMaxAndExp(inputData);
	std::vector<float> subtractMaxAndExpResult = subtractMaxAndExp(program, context, device, queue, INPUT_SIZE, inputForSubtractMaxAndExp, max);
	std::cout << "Subtract Max And Exp - result vector<float> size: " << subtractMaxAndExpResult.size() << std::endl;

	// ---- Step 3. Reduce Sum ----
	std::vector<float> reduceSumResult = reduceSum(program, context, device, queue, INPUT_SIZE, subtractMaxAndExpResult);

	float sum = 0.f;
	if (reduceSumResult.size() > 1) {
		std::vector<float> outSumCopy(reduceSumResult);
		sum = std::accumulate(outSumCopy.begin(), outSumCopy.end(), 0);
		std::cout << "Reduce Sum - sum: " << sum << std::endl;
	} 
	else if (reduceSumResult.size() == 1) 
	{
		sum = reduceSumResult.at(0);
		std::cout << "Reduce Sum - sum: " << sum << std::endl;
	}
	else 
	{
		std::cout << "ERROR: Reduce Sum return nothing!" << std::endl;
	}

	// ---- Step 4. Divide Exponentials by Sum ----
	std::vector<float> divideExpBySumResult = divideExpBySum(program, context, device, queue, INPUT_SIZE, subtractMaxAndExpResult, sum);
	std::cout << "Divide Exponentials by Sum - Done" << std::endl;

	std::vector<float> resultSoftmaxCopy(divideExpBySumResult);
	auto resultSoftmax = std::accumulate(resultSoftmaxCopy.begin(), resultSoftmaxCopy.end(), 1);
	std::cout << "Verify softmax validity(should sum up to 1): " << resultSoftmax << std::endl;
	
	return resultSoftmaxCopy;
}

int main()
{
	auto program = CreateProgram("kernels.cl");
	auto context = program.getInfo<CL_PROGRAM_CONTEXT>();
	auto devices = context.getInfo<CL_CONTEXT_DEVICES>();
	auto& device = devices.front();

	const int INPUT_SIZE = 512;

	cl::CommandQueue queue(context, device);

	// Generate random or fixed input data
	std::vector<float> inputData = generateInputData(INPUT_SIZE, -5.0f, 5.0f); //generateFixedInputData(INPUT_SIZE); // Alternative: std::vector<float> inputData = generateInputData(INPUT_SIZE, -5.0f, 5.0f);
	std::vector<float> outputData = softmax(program, context, device, queue, INPUT_SIZE, inputData);
	std::cout << "Done..." << std::endl;
}