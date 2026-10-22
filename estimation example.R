library(SuperLearner)
library(caret)
library(ranger)
library(glmnet)
library(purrr)
library(dplyr)
library(extraDistr)

dag_integration <- function(num_M, num_Z, W_vars, data_dag,
                            super_learner_Y, super_learner_M, super_learner_Z) {
  # data_dag has A = 0/1 fixed 
  # SL_Y fitted on S = 1, SL_M fitted on S = 0, SL_Z fitted on S = 0
  
  n <- nrow(data_dag)
  h <- rep(0, n)
  
  for (j in 1:num_Z) {
    z_val <- j - 1
    
    pz1 <- as.numeric(
      predict(super_learner_Z,
              newdata = data_dag[, c("A", W_vars), drop = FALSE])$pred
    )
    pz <- if (z_val == 1) pz1 else 1 - pz1
    
    # predict M fixing the corresponding z
    data_m <- data.frame(A = data_dag$A, Z = z_val, data_dag[, W_vars, drop = FALSE])
    pm1 <- as.numeric(predict(super_learner_M, newdata = data_m)$pred)
    
    for (i in 1:num_M) {
      m_val <- i - 1
      pm <- if (m_val == 1) pm1 else 1 - pm1
      
      data_mz <- data.frame(A = data_dag$A, M = m_val, Z = z_val, data_dag[, W_vars, drop = FALSE])
      mu_y <- as.numeric(predict(super_learner_Y, newdata = data_mz)$pred)
      
      h <- h + pz * pm * mu_y
    }
  }
  
  return(h)
}
h1_dr_func = function(data_dag, e_A, e_S, m1, q_S1, h1){
  C1_h  =  (data_dag$A * data_dag$S / (e_A * (1 - e_S))) * ((1 - q_S1) / q_S1) * (data_dag$Y - m1)
  C2_h  =   (data_dag$A * (1-data_dag$S) / (e_A * (1 - e_S))) * (m1 - h1)
  h1_dr =  C1_h + C2_h + h1
  return(h1_dr)
}
h0_dr_func = function(data_dag, e_A, e_S, m0, q_S0, h0){
  C1_h  =  ((1-data_dag$A) * data_dag$S / ((1 - e_A) * (1 - e_S))) * ((1 - q_S0) / q_S0) * (data_dag$Y - m0)
  C2_h  =   ((1-data_dag$A) * (1-data_dag$S) / ((1 - e_A) * (1 - e_S))) * (m0 - h0)
  h0_dr  =  C1_h + C2_h + h0
  return(h0_dr)
}


trans_mediator_one_step = function(data_dag, A, M, Z, S, W_vars, V, Y, learners, learners_single_covariate){
  
  data_dag <- data_dag %>%
    dplyr::rename(
      A = A,
      M = M,
      Z = Z,
      S = S,
      Y = Y
    )
  
  
  # Make a copy for data
  n  =  nrow(data_dag)
  num_M = length(unique(data_dag$M))
  num_Z = length(unique(data_dag$Z))
  data_dag_saved  =  data_dag
  
  # Regression for Y
  data_S0 = data_dag %>% filter (S == 0)
  data_S1 = data_dag %>% filter (S == 1)
  
  # Fit the Super Learner model on S == 1
  super_learner_Y = SuperLearner(Y = data_S1$Y, X = data_S1[, c("A", "M", "Z", W_vars), drop = FALSE],
                                 family = gaussian(),
                                 SL.library = learners,
                                 method = "method.NNLS")
  
  # Prepare the full dataset for prediction
  data_dag_A1 <- data_dag %>% mutate(A = 1)
  data_dag_A0 <- data_dag %>% mutate(A = 0)
  m = predict(super_learner_Y, newdata = data_dag[, c("A", "M", "Z", W_vars), drop = FALSE])$pred
  m1 = predict(super_learner_Y, newdata = data_dag_A1[, c("A", "M", "Z", W_vars), drop = FALSE])$pred
  m0 = predict(super_learner_Y, newdata =  data_dag_A0[, c("A", "M", "Z", W_vars), drop = FALSE])$pred
  
  # e_S = P(S = 1 | W = w)
  data_dag = data_dag_saved
  W_full =  data_dag[, W_vars, drop = FALSE]
  super_learner_S = SuperLearner(Y = data_dag$S, X = W_full,
                                 family = binomial(),
                                 SL.library = learners,
                                 method = "method.NNLS")
  
  pred_S = predict(super_learner_S, newdata = W_full)$pred
  e_S = pred_S[,1]
  
  
  # P(M | Z, A, W, S = 0)
  super_learner_M <- SuperLearner(
    Y = data_S0$M,
    X = data_S0[, c("A", "Z", W_vars), drop = FALSE],
    family = binomial(),
    SL.library = learners,
    method = "method.NNLS"
  )
  
  # P(Z | A, W, S = 0)
  super_learner_Z <- SuperLearner(
    Y = data_S0$Z,
    X = data_S0[, c("A", W_vars), drop = FALSE],
    family = binomial(),
    SL.library = learners,
    method = "method.NNLS"
  )
  
  # h(A, W) - observed A
  h  <- dag_integration(num_M, num_Z, W_vars, data_dag,  super_learner_Y, super_learner_M, super_learner_Z)
  h1 <- dag_integration(num_M, num_Z, W_vars, data_dag_A1, super_learner_Y, super_learner_M, super_learner_Z)
  h0 <- dag_integration(num_M, num_Z, W_vars, data_dag_A0, super_learner_Y, super_learner_M, super_learner_Z)
  
  # q_S = P(S = 1 | A, W, X)
  data_dag = data_dag_saved
  X_full = data_dag[, c("A", "Z", "M", W_vars), drop = FALSE]
  super_learner_S = SuperLearner(Y = data_dag$S, X = X_full,
                                 family = binomial(),
                                 SL.library = learners,
                                 method = "method.NNLS")
  
  pred_S = predict(super_learner_S, newdata = X_full)$pred
  q_S = pred_S[,1]
  
  X_A1 =  data_dag_A1[, c("A", "Z", "M", W_vars), drop = FALSE]
  pred_S = predict(super_learner_S, newdata = X_A1)$pred
  q_S1 = pred_S[,1]
  
  X_A0 =  data_dag_A0[, c("A", "Z", "M", W_vars), drop = FALSE]
  pred_S = predict(super_learner_S, newdata = X_A0)$pred
  q_S0 = pred_S[,1]
  
  # e_A = P(A = 1 | W, S = 0)
  super_learner_e_A <- SuperLearner(
    Y = data_S0$A,
    X = data_S0[, W_vars, drop = FALSE],
    family = binomial(),
    SL.library = learners,
    method = "method.NNLS"
  )
  
  e_A <- as.numeric(predict(super_learner_e_A, newdata = data_dag[, W_vars, drop = FALSE])$pred)
  
  # P(S = 0 | V)
  super_learner_S_V = SuperLearner(Y = data_dag$S, X = data_dag[, c(V), drop = FALSE],
                                   family = binomial(),
                                   SL.library =  learners_single_covariate,
                                   method = "method.NNLS")
  pred_S = predict(super_learner_S_V, newdata = data_dag[, c(V), drop = FALSE])$pred
  prob_S0_cond_V = 1-pred_S[,1]
  
  prob_S0 = 1 - mean(data_dag_saved$S)
  
  h1_eif = h1_dr_func(data_dag, e_A, e_S, m1, q_S1, h1)
  h0_eif = h0_dr_func(data_dag, e_A, e_S, m0, q_S0, h0)
  
  X_full =  data_dag[, c(W_vars), drop = FALSE]
  super_learner_h1 = SuperLearner(Y = h1_eif, X = X_full, 
                                  family = gaussian(),
                                  SL.library = learners,
                                  method = "method.NNLS")
  #   X_full$S = rep(1,n)
  h1_DR = predict(super_learner_h1, newdata = X_full)$pred
  
  
  super_learner_h0 = SuperLearner(Y = h0_eif, X = X_full, 
                                  family = gaussian(),
                                  SL.library = learners,
                                  method = "method.NNLS")
  #  X_full$S = rep(1,n)
  h0_DR = predict(super_learner_h0, newdata = X_full)$pred
  
  f_DR = h1_DR - h0_DR
  g_DR = h0_DR
  
  h_DR = data_dag$A * f_DR + g_DR
  
  temp = data.frame(data_dag$S, f_DR)
  theta_plug_in = temp %>% 
    filter(data_dag.S == 0) %>% 
    summarise(mean_f = mean(f_DR)) %>% 
    pull(mean_f) # plug-in for theta
  
  
  # P(S = 0 | V)
  
  data_dag = data_dag_saved
  super_learner_S_V = SuperLearner(Y = data_dag$S, X = data_dag[, c(V), drop = FALSE],
                                   family = binomial(),
                                   SL.library =  learners_single_covariate,
                                   method = "method.NNLS")
  pred_S = predict(super_learner_S_V, newdata = data_dag[, c(V), drop = FALSE])$pred
  prob_S0_cond_V = 1-pred_S[,1]
  
  # E(f | V)
  data_dag = data_dag_saved
  super_learner_f = SuperLearner(Y = f_DR, X = data_dag[, c(V), drop = FALSE],
                                 family = gaussian(),
                                 SL.library = learners_single_covariate,
                                 method = "method.NNLS")
  
  pred_f = predict(super_learner_f, data_dag[, c(V), drop = FALSE])$pred
  #lambda
  data_dag = data_dag_saved
  index_0 = which(data_dag$S == 0)
  lambda_plug_in = mean(pred_f[index_0]) # plug-in for lambda
  
  
  # k = E(f | e_S)
  super_learner_k <- SuperLearner(
    Y = f_DR,
    X = data.frame(e_S = e_S),
    family = gaussian(),
    SL.library = learners_single_covariate,
    method = "method.NNLS"
  )
  k <- predict(super_learner_k, newdata = data.frame(e_S = e_S))$pred
  
  index_0 <- which(data_dag$S == 0)
  tau_plug_in <- mean(k[index_0]) # plug-in for tau
  
  prob_S0 = 1 - mean(data_dag_saved$S)
  
  
  C1 = ((data_dag$S * (data_dag$A - e_A)) / (e_A * (1 - e_A))) * ((1 - q_S) / q_S) * (data_dag$Y - m)
  
  C2 = (((1 - data_dag$S) * (data_dag$A - e_A)) / (e_A * (1 - e_A))) * (m - h_DR)
  
  C3 = (1 - data_dag$S) * (f_DR - theta_plug_in) #update
  
  C4 = prob_S0_cond_V * (f_DR - pred_f)
  
  C5 =  (1 - data_dag$S) * (pred_f - lambda_plug_in)
  
  D_theta = 1/prob_S0 * (C1 + C2 + C3)
  theta_hat = theta_plug_in + mean(D_theta)
  
  D_lambda = (1/prob_S0) * (C1 + C2 + C4 + C5)
  lambda_hat = lambda_plug_in + mean(D_lambda)
  
  
  super_learner_S_m_init <- SuperLearner(
    Y = data_dag$S,
    X = data.frame(m_hat = m),
    family = binomial(),
    SL.library = learners_single_covariate,
    method = "method.NNLS"
  )
  p_S_init <- predict(
    super_learner_S_m_init,
    newdata = data.frame(m_hat = m)
  )$pred[, 1]
  p_S_init <- pmin(pmax(p_S_init, 1e-2), 1 - 1e-2)
  
  
  C1_tau <- ((data_dag$S * (data_dag$A - e_A)) / (e_A * (1 - e_A))) *
    ((1 - p_S_init) / p_S_init) *
    (data_dag$Y - m)
  
  C2_tau <- (((1 - data_dag$S) * (data_dag$A - e_A)) / (e_A * (1 - e_A))) *
    (m - h_DR)
  
  C3_tau <- (1 - e_S) * (f_DR - k)
  
  C4_tau <- (1 - data_dag$S) * (k - tau_plug_in)
  
  D_tau <- (1 / prob_S0) * (C1_tau + C2_tau + C3_tau + C4_tau)
  
  tau_hat <- tau_plug_in + mean(D_tau)
  
  
  return(c(theta_hat, theta_hat - 1.96 * sd(D_theta) / sqrt(n), theta_hat + 1.96 * sd(D_theta)/sqrt(n), 
           lambda_hat, lambda_hat - 1.96 * sd(D_lambda)/sqrt(n), lambda_hat + 1.96 * sd(D_lambda)/sqrt(n),
           tau_hat, tau_hat - 1.96 * sd(D_tau) / sqrt(n), tau_hat + 1.96 * sd(D_tau)/sqrt(n)))
}

# Prevent multithreading issues
Sys.setenv(OMP_NUM_THREADS = "1")
Sys.setenv(OPENBLAS_NUM_THREADS = "1")
Sys.setenv(MKL_NUM_THREADS = "1")

rep_id <- as.integer(Sys.getenv("SLURM_ARRAY_TASK_ID"))
set.seed(1000 + rep_id)

extract_estimates <- function(out) {
  out <- as.numeric(out)
  tibble::tibble(
    theta = out[1],
    theta_l = out[2],
    theta_u = out[3],
    lambda = out[4],
    lambda_l = out[5],
    lambda_u = out[6],
    tau = out[7],
    tau_l = out[8],
    tau_u = out[9]
  )
}

# sample_sizes <- c(500, 1000, 5000, 10000)
sample_sizes <- c(5000)

run_sim <- function(n, rep_id) {
  # -----------------------------
  # 1. Generate data
  # -----------------------------
  # S  <- as.numeric(rbernoulli(n, p = 0.58))
  # W2 <- as.numeric(rbernoulli(n, plogis(2 * S - 1)))
  # W1 <- as.numeric(rbernoulli(n, plogis(2 * W2 + 0.7)))
  # A  <- as.numeric(rbernoulli(n, plogis(0.2*S + 0.2 * W1 + 0.2 * W2 - 0.3)))
  # Z  <- as.numeric(rbernoulli(n, plogis(0.3 * A + 0.4 * W1 + 0.3 * W2 - 0.7)))
  # M  <- as.numeric(rbernoulli(n, plogis(0.3 * A + 0.6 * W1 + 0.25 * W2 - 0.55 * S + cos(Z) - 2)))
  # 
  # data_dag <- data.frame(
  #   S = S, W1 = W1, W2 = W2, A = A, Z = Z, M = M, Y = NA_real_
  # )
  # 
  # data_dag$Y <- rnorm(
  #   n,
  #   mean = 0.4 * W1 - 0.3 * A + 7 * M + 0.33 * Z + 0.92 * W2,
  #   sd = 0.5
  # )
  # 
  # data_dag$Y <- dplyr::case_when(
  #   S == 1 ~ data_dag$Y,
  #   TRUE ~ 0
  # )
  
  S  <- as.numeric(rbernoulli(n, p = 0.58))
  W2 <- as.numeric(rbernoulli(n, plogis(2 * S - 1)))
  W1 <- as.numeric(rbernoulli(n, plogis(2 * W2 + 0.7)))
  A  <- as.numeric(rbernoulli(n, plogis(0.2*S + 0.2 * W1 + 0.2 * W2 - 0.3)))
  Z  <- as.numeric(rbernoulli(n, plogis(0.3 * A + 0.4 * W1 + 0.3 * W2 - 0.7)))
  M  <- as.numeric(rbernoulli(n, plogis(0.3 * A + 0.6 * W1 + 0.25 * W2 - 2 * S + cos(Z) - 2)))
  
  data_dag <- data.frame(
    S = S, W1 = W1, W2 = W2, A = A, Z = Z, M = M, Y = NA_real_
  )
  
  data_dag$Y <- rnorm(
    n,
    mean = 0.4 * W1 - 0.3 * A + 7 * M + 0.33 * Z + 0.92 * W2,
    sd = 0.5
  )
  
  data_dag$Y <- dplyr::case_when(
    S == 1 ~ data_dag$Y,
    TRUE ~ 0
  )
  
  
  
  
  # -----------------------------
  # 2. Learners
  # -----------------------------
  l1 <- c(
    "SL.mean",
    "SL.glm",
    "SL.glmnet",
    "SL.gam",
    "SL.earth",
    "SL.glm.interaction"
    # "SL.ranger",
    # "SL.xgboost"
  )
  
  l2 <- c(
    "SL.mean",
    "SL.glm",
    "SL.gam",
    "SL.earth",
    "SL.glm.interaction"
    # "SL.ranger",
    # "SL.xgboost"
  )
  
  # -----------------------------
  # 3. Methods
  # -----------------------------
  method_list <- list(
    os = trans_mediator_one_step
  )
  
  safe_run <- function(fun, method_name) {
    tryCatch({
      out <- fun(
        data_dag = data_dag,
        A = "A", M = "M", Z = "Z", S = "S",
        W_vars = c("W1", "W2"),
        V = "W2",
        Y = "Y",
        learners = l1,
        learners_single_covariate = l2
      )
      extract_estimates(out)
    }, error = function(e) {
      message(
        "Estimator failed: method = ", method_name,
        ", n = ", n,
        ", rep = ", rep_id,
        ": ", e$message
      )
      tibble::tibble(
        theta = NA_real_, theta_l = NA_real_, theta_u = NA_real_,
        lambda = NA_real_, lambda_l = NA_real_, lambda_u = NA_real_,
        tau = NA_real_, tau_l = NA_real_, tau_u = NA_real_
      )
    })
  }
  
  res_list <- purrr::imap(method_list, ~ safe_run(.x, .y))
  
  # -----------------------------
  # 4. Return tidy row
  # -----------------------------
  dplyr::bind_cols(
    tibble::tibble(
      sample_size = n,
      rep_id = rep_id
    ),
    purrr::imap_dfc(res_list, function(df, nm) {
      names(df) <- paste0(names(df), "_", nm)
      df
    })
  )
}

all_results <- purrr::map_dfr(sample_sizes, ~ run_sim(.x, rep_id))

saveRDS(
  all_results,
  file = paste0(
    "/gpfs/home/userid/transport_mediator/result/sim_job_",
    Sys.getenv("SLURM_ARRAY_JOB_ID"),
    "_rep_",
    rep_id,
    ".rds"
  )
)

