build_cpu:
	docker build -t ft-data-gen:cpu .
build_gpu:
	docker build -t ft-data-gen:gpu -f Dockerfile.gpu .
run_cpu_container:
	docker run -it --rm -v ${PWD}:/workspace ft-data-gen:cpu bash
run_gpu_container:
	docker run -it --rm -v ${PWD}:/workspace --gpus all ft-data-gen:gpu
generate_data_test:
	python generate_data.py --input_file data/test.txt --qa_num 2
