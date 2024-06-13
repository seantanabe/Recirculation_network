

clear all
logistic = @(x)(1./(1 + exp(-x)));

% param
N_hid = 2; % # hidden units
N_vis = 4; % # visible units
vis_vec = [1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1]; % visible vectors
vis_vec = [vis_vec ones(4,1)]; % eliminate theta
lambda = 0.75; % lambda
epsi = 20;     % epsilon
E_stop = 0.1;  % stop if all error<E_stop
N_sim = 100;   % # simulations
N_train_max  = 400; % # train/loops/learn within 1 simulation
N_train_stop = NaN(N_sim,1); % record # training till all error<E_stop

y_vis_i = ones(N_vis + 1,1);
x_vis_i = NaN(N_vis,1);
y_hid_i = ones(1,N_hid + 1);
x_hid_i = NaN(1,N_hid );
for n_i = 1:N_sim
    E_train = NaN(N_vis,N_train_max);
    % assuming weights are seperate for feedforward/feedback
    w_vis_hid = rand(N_vis + 1, N_hid)-0.5; % weights, visible units to hidden
    w_hid_vis = rand(N_hid + 1, N_vis)-0.5; % weights, hidden units to visible
    for t_i = 1:N_train_max
        disp(['simulation ' num2str(n_i) ' training ' num2str(t_i)])
        w_vis_hid_hold = zeros(size(w_vis_hid));
        w_hid_vis_hold = zeros(size(w_hid_vis));
        for v_i = 1:size(vis_vec,1)
            % time = 0
            y_vis_i = vis_vec(v_i,:)';
            y_vis_0 = y_vis_i; % save state
            
            % time = 1
            x_hid_i           = y_vis_i'*w_vis_hid;
            y_hid_i(1:N_hid)  = logistic(x_hid_i);
            y_hid_1           = y_hid_i; % save state
            
            % time = 2
            x_vis_i          = w_hid_vis'*y_hid_i';
            y_vis_2_lam0     = [logistic(x_vis_i); 1]; % save state, lambda = 0, extra input '1'
            y_vis_i(1:N_vis) = lambda*y_vis_0(1:N_vis) + (1-lambda)*logistic(x_vis_i); % this logistic(x) wasn't in paper
            y_vis_2          = y_vis_i; % save state
            
            % time = 3
            x_hid_i          = y_vis_i'*w_vis_hid;
            y_hid_i(1:N_hid) = lambda*y_hid_1(1:N_hid) + (1-lambda)*logistic(x_hid_i);
            y_hid_3          = y_hid_i; % save state
            
            % hold weight change
            w_tmp          = epsi*(y_hid_1.*(y_vis_0 - y_vis_2))';
            w_hid_vis_hold = w_hid_vis_hold + w_tmp(:,1:end-1);
            w_tmp          = epsi*y_vis_2.*(y_hid_1 - y_hid_3);
            w_vis_hid_hold = w_vis_hid_hold + w_tmp(:,1:end-1);
            
            E                = 0.5*sum((y_vis_2_lam0 - y_vis_0).^2);
            E_train(v_i,t_i) = E;
        end
        if sum(E_train(:,t_i) < E_stop) == size(vis_vec,1)
            N_train_stop(n_i) = t_i - 1;
            break
        end
        % update weights
        w_hid_vis = w_hid_vis + w_hid_vis_hold;
        w_vis_hid = w_vis_hid + w_vis_hid_hold;
    end
end



figure('Renderer', 'painters', 'Position', [10 10 900 300])
subplot(1,2,1)
histogram(N_train_stop,10,'FaceColor','k')
hold on
line(nanmean(N_train_stop)*[1 1],ylim,'Color','red','LineWidth',1.2)
xlabel({'Training number required' ['for reconstruction error < ' num2str(E_stop)]})
ylabel('Simulation count')
title({'Training number till learned', ['N_{simulation} = ' num2str(N_sim) ', red line is mean']})
box off

subplot(1,2,2)
imagesc(E_train(:,1:(t_i)))
xlabel('Training number')
ylabel('Training cases')
h = colorbar;
ylabel(h, 'Reconstruction error')
colormap(flip(gray))
title({'Training error example',['Simulation #' num2str(n_i)]})

