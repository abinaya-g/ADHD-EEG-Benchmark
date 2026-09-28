function X = initialization(N, dim, up, down)
    % Initialize a random matrix inside the provided bounds.
    % N (integer): number of rows.
    % dim (integer): number of dimensions.
    % up (vector/scalar): upper bound.
    % down (vector/scalar): lower bound.
    
    if numel(up) == 1
        X = rand(N, dim) .* (up - down) + down;
    else
        X = zeros(N, dim);
        for i = 1:dim
            high = up(i);
            low = down(i);
            X(:, i) = rand(N, 1) .* (high - low) + low;
        end
    end
end
