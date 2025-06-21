Las operaciones clave paralelizadas incluyen la propagación hacia adelante (forward_layer_kernel), 
el cálculo de deltas en la capa de salida (calculate_output_deltas_kernel), 
la retropropagación de deltas a capas anteriores (backpropagate_deltas_kernel), y 
la actualización de pesos y sesgos (update_weights_biases_kernel), donde cada hilo CUDA maneja el procesamiento de neuronas individuales o sus parámetros. 

El programa principal (main.cpp) orquesta la inicialización (create_mlp_cuda), 
el entrenamiento por muestras (train_sample_cuda), 
la predicción (predict_mlp_cuda), y 
la liberación de memoria (destroy_mlp_cuda), 
generando logs de pérdida y una matriz de confusión para evaluar el rendimiento.
