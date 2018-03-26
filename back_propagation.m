% Neural Network - Back Propagation algorithm.
function [ output_args ] = back_propagation( training_file, test_file, num_layers, units_per_layer, rounds )

    file_data = load (training_file);
    [r, c] = size(file_data);
    
    training_data = sortrows(file_data, c);
    
    [num_classes, max_class, start_index, end_index, t] = get_start_end_index(training_data);
    
    [x_input, target] = normalize_and_add_bias (training_data);
    
    weight_map = containers.Map('KeyType','int32','ValueType','any');
    learning_rate = 1;
    
    if num_layers <= 2
            units_per_layer = num_classes;
    end
    rows_per_class = units_per_layer / num_classes;
    
    [r, c] = size(x_input);
    r_input = zeros(length(start_index) .* rows_per_class, c);
    for round_count=1:rounds
        
        s_index = start_index;
        for x=1:(max_class/rows_per_class)
        
            [r_input, s_index] = first_layer_input(x_input, r_input, s_index, end_index, rows_per_class);

            [z_input, weight_map, z_map, l] = calculate_output(r_input, weight_map, num_layers, rows_per_class, num_classes, round_count);

            %error = t - z_input;
            %disp (round_count);

            d_output = (z_input - t) .* z_input .* (1 - z_input);
            d_output = expand(d_output, rows_per_class);
            %d_z = learning_rate * (d_output .* z_map(l - 1));
            z_input = z_map(l - 1);
            d_z = learning_rate * (d_output .* z_input);
            if num_layers > 2
                d_z = reshape(d_z, rows_per_class, num_classes)';
            end
            %d_z = reshape(learning_rate * (d_output .* z_map(l - 1)), num_classes, rows_per_class);
            %d_z = d_z, rows_per_class, num_classes)';
            weights = weight_map(num_layers);
            weights = weights - d_z;
            weight_map(l) = weights;

            for i=(num_layers - 1):-1:2

                weights = weight_map(i + 1);

                if i == (num_layers - 1)
                    d_output = reshape(d_output, rows_per_class, num_classes)';
                end
                d_output = d_output .* weights;

                z_input = z_map(i);
                [r, c] = size(z_input);
                d_output = reshape(d_output', r, c); % Reshape
                d_hidden = d_output .* z_input .* (1 - z_input);

                z_input = z_map(i - 1);
                weights = weight_map(i);
                weights = weights - learning_rate .* d_hidden .* z_input;
                weight_map(i) = weights;
                d_output = d_hidden;
            end
                        
        end
        
        learning_rate = learning_rate * 0.98;
        
        %disp(weight_map(2));
        %disp(weight_map(3));
    end
    
    classify(test_file, weight_map, num_layers, rows_per_class, num_classes);
        
end

function [] = classify(test_file, weight_map, num_layers, rows_per_class, num_classes)

    test_data = load (test_file);    
    
    [x_input, target] = normalize_and_add_bias (test_data);
    
    [r, c] = size(x_input);
    
    cum_accuracy = 0.0;
    for i=1:r
        
        row = x_input(i,:);
        z_input = repmat(row, num_classes * rows_per_class, 1);
        t = target(i,1);
        
        [z_input, weight_map, z_map, l] = calculate_output(z_input, weight_map, num_layers, rows_per_class, num_classes, -1);
        
        [max_val, index] = max(z_input);
        max_index = find (z_input == max_val);
        
        if length(max_index) > 1
            t_index = find (max_index == t);
            if ~isempty(t_index)
                accuracy = 1/length(max_index);
            end
        else 
            if (max_index - 1) == t
                accuracy = 1;
            else
                accuracy = 0;
            end
        end
        
        cum_accuracy = cum_accuracy + accuracy;
        
        fprintf('ID=%5d, predicted=%3d, true=%3d, accuracy=%4.2f\n', i, index - 1, t, accuracy);
    end
    
    fprintf ('classification accuracy=%6.4f\n', (cum_accuracy / r));
end

function [z_input, weight_map, z_map, l] = calculate_output(z_input, weight_map, num_layers, rows_per_class, num_classes, round_count)
    
    z_map = containers.Map('KeyType','int32','ValueType','any');
    z_map(1) = z_input; % Hardcode layer 1
    
    if num_layers > 2

        for l=2:(num_layers - 1)

            if round_count == 1
                weights = getWeights(size(z_input));
                weight_map(l) = weights;
            else
                weights = weight_map(l);
            end

            %a = sum(weights .* z_input, 2); %sigmoid(a);
            z_input = layer_output(weights, z_input);
            z_map(l) = z_input;
        end
    
    end
    
    l = num_layers;
    if round_count == 1
        %weights = getWeights([num_classes, rows_per_class]);
        weights = getWeights(size(z_input));
        if num_layers > 2
            weights = reshape(weights, rows_per_class, num_classes)';
        end
        weight_map(l) = weights;
    else
        weights = weight_map(l);
    end
    
    if num_layers > 2
        z_input = reshape(z_input, rows_per_class, num_classes)';        
    end
    z_input = layer_output(weights, z_input);                   % Output of last layer
    z_map(l) = z_input;
end

function [d] = expand(d_output, rows_per_class)
    
    d = []
    for i=1:length(d_output)
        for j=1:rows_per_class
            d = [d d_output(i,:)];
        end
    end
    
    d = d';
end
        
function [z_input] = layer_output(weights, z_input)
    a = sum(weights .* z_input, 2);
    z_input = sigmoid(a);
end

function [z_input] = sigmoid(a)
    z_input = zeros(length(a), 1);
    for i=1:length(a)
        z_input(i, 1) = 1.0/(1.0 + exp(-a(i,:)));
    end
end

function [weights] = getWeights(size)
    high = 0.05; low = -0.05;
    weights = rand(size(1, 1), size(1, 2)) * (high - low) + low;
end    

function [z_input, start_index] = first_layer_input(x_input, r_input, start_index, end_index, rows_per_class)
    
    %[r, c] = size (x_input);
    
    z_input = r_input;
    
    %t = zeros(rows_per_class * length(start_index), 1);    
    %t = zeros(rows_per_class * length(start_index), 1);
    index = 1;
    for i=1:length(start_index)
        for j=start_index(i):(start_index(i) + rows_per_class - 1)
            if j > end_index(i)
                break;
            end
            z_input(index,:) = x_input(j,:);
            %if ~isempty(find(t == target(j,:)))
            %    t(index,:) = target(j,:);
            %end
            index = index + 1;
            start_index(i) = start_index(i) + 1;
        end
    end
end

function [num_classes, max_class, start_index, end_index, classes] = get_start_end_index (training_data)
    
    [r, c] = size(training_data);
    classes = unique(training_data(:,c));
    num_classes = length(classes);
    num_each_class = histc(training_data(:,c), classes);
    max_class = max(num_each_class);
    end_index = cumsum(num_each_class);
    end_index = end_index';
    start_index = ones(1, length(end_index));
    for i =1:length(end_index) - 1
        start_index(i + 1) = end_index(i) + 1;
    end
end

function [x_input, target] = normalize_and_add_bias (training_data)

    [r, c] = size (training_data);
    x_input = training_data (:, 1:c-1);
    target = training_data (:, c);

    max_x = max(x_input);
    max_x = max(max_x);             % Max Number

    x_input = x_input / max_x;      % normalization

    x_input = [ones(r, 1) x_input];
end
