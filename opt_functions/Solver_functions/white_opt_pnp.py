import os
from . import *
import numpy as np
import torch
from tqdm.auto import tqdm
from deepinv.physics import Denoising, GaussianNoise, PoissonNoise
from deepinv.utils.demo import load_url_image, get_image_url
from deepinv.utils.plotting import plot
from skimage.metrics import structural_similarity
import torchmin
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import math
from .projected_gradient import *

def pnp_ism(y, back, parameters_pnp, device):
    
    data_fid = parameters_pnp["data_fid"]
    grad_data_fid  = parameters_pnp["grad_data_fid"] 
    single_fid  = parameters_pnp["single_data_fid"] 
    tollerance  = parameters_pnp["tollerance"]      
    max_iter  = parameters_pnp["max_iter"]     
    x_init  = parameters_pnp["x_init"]     
    sigma = parameters_pnp["sigma"]  
    L_max  = parameters_pnp["Lip_reg"]      
    pnp   = parameters_pnp["Pnp"]  
    physics   = parameters_pnp["physics"]
    

    x_gt = parameters_pnp['ground_truth']
    # Sposto tutti i tensori sul device
    funct = torch.zeros(max_iter, device=device)
    iter_err = torch.zeros(max_iter, device=device)
    norm2 = torch.zeros(max_iter, device=device)
    psnr_vec = torch.zeros(max_iter, device=device)
    ssim_vec = torch.zeros(max_iter, device=device)
    
    
    min_distance = - float('inf')

    x_k_prec = x_init.to(device)  # Assicurati che x_init sia sul device
    y = y.to(device)
    back = back.to(device)

    tau = 1/L_max
    # for k in tqdm(range(max_iter), desc="Iterations") :
    for k in range(max_iter):
        with torch.no_grad():

            x_k_succ = torch.max(x_k_prec -  tau* grad_data_fid(y, x_k_prec,  physics), torch.tensor(0))
            x_k_succ = x_k_succ/ x_k_succ.max()
            # x_k_succ[:,1:2] = pnp(x_k_succ[:,1:2], sigma)
            x_k_succ = pnp(x_k_succ, sigma)
            x_k_succ = torch.clamp(x_k_succ, 0, 1)
                
        funct[k] = data_fid(y, x_k_succ, physics)
        norm2[k] = torch.norm(x_gt - x_k_succ, 'fro')
        iter_err[k] = torch.norm(x_k_prec - x_k_succ, 'fro') / torch.norm(x_k_prec, 'fro')
        psnr_vec[k] = psnr(x_gt/x_gt.max(), x_k_succ/x_k_succ.max())
        ssim_vec[k] = ssim(x_gt/x_gt.max(), x_k_succ/x_k_succ.max())
                        
        
        if iter_err[k] < tollerance:
            print(f"Convergence reached at iteration = {k}")
            funct = funct[0:k]
            iter_err = iter_err[0:k]
            norm2 = norm2[0:k]
            # lpips_vec = lpips_vec[0:k] 
            break

        x_k_prec = x_k_succ
        

    return x_k_succ, funct.detach(), iter_err.detach(), norm2.detach(), psnr_vec.detach(), ssim_vec.detach()


def standardize(Y, lam):
    """
    lam: scalare (float o tensor)
    """
    lam = torch.as_tensor(lam, device=Y.device, dtype=Y.dtype)
    Z = (Y - lam) / torch.sqrt(lam)
    return Z


def T_fun(lam):
    """Calcola T(λ) = 1 / (1 - exp(-λ)) in float64"""
    # Cast a float64 per il calcolo interno
    lam64 = lam.to(torch.float64)
    
    # Calcolo con maggiore precisione
    res = 1.0 / (1.0 - torch.exp(-lam64))
    
    # Ritorna in float32
    return res.to(torch.float32)

def V_fun(lam):
    """Calcola V(λ) in float64"""
    lam64 = lam.to(torch.float64)
    exp_neg_lam = torch.exp(-lam64)
    
    # Formula: (1 - (1+λ)exp(-λ)) / (1 - exp(-λ))^2
    num = 1.0 - (1.0 + lam64) * exp_neg_lam +1e-10
    den = (1.0 - exp_neg_lam)**2 + 2e-10
    
    # Nota: Rimosso l'epsilon manuale poiché float64 gestisce meglio i valori piccoli.
    # Se lam è esattamente 0, den sarà 0.
    res = num / den
    
    return res.to(torch.float32)



def compute_truncation_stats_eps(lam, eps):
    """
    Calcola Valore Atteso e Varianza vettorizzando le operazioni per la massima velocità su GPU.
    """
    lam64 = lam.to(torch.float64)
    
    # Evitiamo log(0) mettendo un limite inferiore infinitesimo a lambda
    lam_safe = torch.clamp(lam64, min=1e-12) 
    
    k = int(math.floor(eps))
    
    # 1. Creiamo un tensore z con tutti i valori da 0 a k direttamente sulla stessa GPU di lam
    z = torch.arange(k + 1, dtype=torch.float64, device=lam.device)
    
    # 2. Aggiungiamo una dimensione fittizia a lam per poter eseguire il "broadcasting"
    # Se lam è shape (Batch,), diventa (Batch, 1) in modo da incrociarsi con z che è shape (k+1,)
    lam_ext = lam_safe.unsqueeze(-1)
    
    # 3. Calcoliamo i logaritmi delle probabilità tutto in una volta
    # torch.lgamma(z + 1) è l'equivalente matematico di ln(z!)
    log_prob_z = -lam_ext + z * torch.log(lam_ext) - torch.lgamma(z + 1.0)
    
    # Torniamo alle probabilità lineari (exp annulla il log)
    prob_z = torch.exp(log_prob_z)
    
    # 4. Calcoliamo le somme (S0, S1, S2) collassando l'ultima dimensione (dim=-1)
    S0 = prob_z.sum(dim=-1)
    S1 = (z * prob_z).sum(dim=-1)
    S2 = ((z ** 2) * prob_z).sum(dim=-1)
    
    # T_eps(lambda) = 1 / (1 - S0)
    T_eps = 1.0 / (1.0 - S0 + 1e-12)
    
    # Valore Atteso
    E_eps = T_eps * (lam64 - S1)
    
    # Varianza
    Var_eps = T_eps * (lam64 * (1.0 + lam64) - S2) - (E_eps ** 2)
    Var_eps = torch.clamp(Var_eps, min=1e-12)
    
    # Rimuoviamo la dimensione fittizia per tornare alla shape originale e passiamo a float32
    return E_eps.squeeze(-1).to(torch.float32), Var_eps.squeeze(-1).to(torch.float32)



def standardize_unbiased_masked(Y, lam):
    """
    Calcola la versione 'masked' z_+(U) come definita in Sezione 4.5.
    
    Y: tensore delle osservazioni (y)
    lam: tensore della media stimata (λ_hat)
    """
    # 1. Calcolo di T e V
    T = T_fun(lam)
    V = V_fun(lam)
    
    # 2. Standardizzazione secondo la formula (28):
    # z(U) = (y - λ*T(λ)) / sqrt(λ*V(λ))
    Z = (Y - lam * T) / torch.sqrt(lam * V)
    
    # 3. Applicazione del blind masking (I_+):
    # Restituisce z_i se i appartiene a I_+, altrimenti 0.
    # Assumendo I_+ come l'indice delle Y positive (y_i > 0).
    Z_plus = torch.where(Y > 0, Z, torch.zeros_like(Z))
    
    return Z_plus


def standardize_unbiased_masked_eps(Y, lam, eps):
    """
    Calcola la versione 'masked' Z_eps per Y > eps.
    
    Y: tensore delle osservazioni (y)
    lam: tensore della media stimata (λ_hat)
    eps: scalare, soglia di troncamento (es. 0, 1.5, 3)
    """
    # 1. Calcolo di Valore Atteso e Varianza troncati
    E_eps, Var_eps = compute_truncation_stats_eps(lam, eps)
    
    # 2. Standardizzazione: Z_eps = (Y - E_eps) / sqrt(Var_eps)
    Z_eps = (Y - E_eps) / torch.sqrt(Var_eps)
    
    # 3. Applicazione del blind masking:
    # Restituisce Z_eps se Y > eps, altrimenti 0.
    Z_eps_masked = torch.where(Y > eps, Z_eps, torch.zeros_like(Z_eps))
    
    return Z_eps_masked

# def whiteness_measure(Z: torch.Tensor) -> torch.Tensor:
#     """
#     Calcola W(Z) utilizzando la Proposizione 2.1 (dominio delle frequenze).
#     Assume condizioni al contorno periodiche.
    
#     Z: tensor 3D (m1, m2, m3), reale.
#     Ritorna: scalare (Tensor) rappresentante la misura di bianchezza.
#     """
#     if Z.ndim != 3:
#         # Se Z ha una dimensione batch o canale (es. 1, m1, m2, m3), la rimuoviamo
#         Z = Z.squeeze()
#         if Z.ndim != 3:
#             raise ValueError(f"Z deve essere 3D, ma ha shape {Z.shape}")

#     # 1. Trasformata di Fourier 3D (non serve padding con condizioni periodiche)
#     # Usiamo rfftn (Real FFT) per efficienza se Z è reale
#     FZ = torch.fft.fftn(Z)
    
#     # 2. Calcolo dei moduli al quadrato |z_tilde|^2
#     # Nota: abs() su un complesso restituisce il modulo, eleviamo alla seconda
#     mag_sq = torch.abs(FZ)**2
    
#     # 3. Numeratore: Somma dei moduli alla quarta power
#     numerator = torch.sum(mag_sq**2)
    
#     # 4. Denominatore: Quadrato della somma dei moduli al quadrato
#     denominator = torch.sum(mag_sq)**2
    
#     if denominator == 0:
#         raise ValueError(f"Denominatore uguale a zero")
        
#     return numerator / denominator


def whiteness_measure(Z: torch.Tensor, mode="standard", cutoff=0.10):
    """
    Calcola W(Z) nel dominio delle frequenze.
    
    Parametri:
    - Z: tensor 3D (m1, m2, m3), il residuo standardizzato.
    - mode: "standard" (tutte le frequenze) o "highpass" (ignora le basse frequenze).
    - cutoff: percentuale della dimensione del tensore da azzerare al centro (es. 0.10 = 10%).
    
    Ritorna: 
    - wh: il valore grezzo del rapporto del Whiteness Principle.
    - M_eff: il numero di frequenze effettivamente utilizzate (serve per la normalizzazione).
    """
    print(Z.shape)
    if Z.ndim != 3:
        Z = Z.squeeze()
        if Z.ndim != 3:
            raise ValueError(f"Z deve essere 3D, ma ha shape {Z.shape}")

    # 1. Trasformata di Fourier 3D
    FZ = torch.fft.fftn(Z)
    
    # 2. Shift: spostiamo la frequenza 0 (DC) esattamente al centro del tensore
    # Questo ci permette di tagliare via le frequenze basse molto più facilmente
    FZ_shifted = torch.fft.fftshift(FZ)
    
    # 3. Spettro di potenza (modulo al quadrato)
    mag_sq = torch.abs(FZ_shifted)**2
    
    # Di default, il numero di frequenze attive è il numero totale di pixel
    M_eff = Z.numel() 
    
    # 4. Applicazione del filtro passa-alto
    if mode == "highpass":
        s0, s1, s2 = mag_sq.shape
        # Coordinate del centro esatto
        c0, c1, c2 = s0 // 2, s1 // 2, s2 // 2
        
        # Calcoliamo quanti pixel tagliare a destra e sinistra del centro
        # (se cutoff è 0.1, tagliamo +/- 5% dal centro)
        r0 = max(1, int(s0 * cutoff / 2))
        r1 = max(1, int(s1 * cutoff / 2))
        r2 = max(1, int(s2 * cutoff / 2))
        
        # Azzera il blocco centrale (cioè cancella le basse frequenze)
        mag_sq[c0-r0:c0+r0, c1-r1:c1+r1, c2-r2:c2+r2] = 0.0
        
        # Aggiorniamo il numero di frequenze attive sottranedo quelle azzerate
        pixel_tagliati = (2*r0) * (2*r1) * (2*r2)
        M_eff = M_eff - pixel_tagliati
    
    # 5. Numeratore e Denominatore
    numerator = torch.sum(mag_sq**2)
    denominator = torch.sum(mag_sq)**2
    
    if denominator == 0:
        raise ValueError("Denominatore uguale a zero: o Z è vuoto, o il taglio è troppo grande.")
        
    return (numerator / denominator), M_eff


def find_knee_point(mu_values, W_sum):
    # Lavoriamo in scala logaritmica per mu se la griglia è logaritmica
    log_mu = np.log10(mu_values + 1e-10)

    # Normalizziamo entrambi gli assi tra 0 e 1 per confrontarli
    w_norm = (W_sum - W_sum.min()) / (W_sum.max() - W_sum.min())
    mu_norm = (log_mu - log_mu.min()) / (log_mu.max() - log_mu.min())

    # Linea che unisce il primo e l'ultimo punto
    # Distanza di ogni punto dalla retta: |(y2-y1)x - (x2-x1)y + x2y1 - y2x1|
    p1 = np.array([mu_norm[0], w_norm[0]])
    p2 = np.array([mu_norm[-1], w_norm[-1]])

    distances = []
    for i in range(len(mu_norm)):
        p3 = np.array([mu_norm[i], w_norm[i]])
        dist = np.cross(p2-p1, p1-p3) / np.linalg.norm(p2-p1)
        distances.append(dist)

    best_idx = np.argmax(distances)
    return mu_values[best_idx]


def RWP_PNP(dataset, parameters, hparams, mask_type="masked", eps_f=0):
    """
    Calcola il Residual Whiteness Principle (RWP) per una griglia di parametri sigma.
    Salva PSNR e SSIM solo per i dati simulati.
    """
    # 1. RECUPERA PARAMETRI E DEVICE
    noise_image = dataset["noise_image"]
    back_vec = dataset["back_vec"]
    
    sigma_values_grid = hparams["sigma_grid"]
    is_3d = hparams['IS_3D']
    is_realdata = hparams['IS_REAL']

    device = noise_image.device

    M = noise_image.numel() 
    n = len(sigma_values_grid)
    
    # Inizializza vettori: validi per dati simulati, None per dati reali
    W_sum = torch.empty(n, device=device)    
    psnr_vecs = torch.empty(n, device=device) if not is_realdata else None 
    ssim_vecs = torch.empty(n, device=device) if not is_realdata else None   
    
    min_distance = float('inf') 
    physics = parameters["physics"]

    # --- 2. PRE-CALCOLO GROUND TRUTH, Z_TRUE E EPSILON ---
    
    # Pre-calcoliamo epsilon in modo sicuro, indipendentemente se i dati sono reali o simulati
    if mask_type == "masked_eps":
        if is_3d:
            # Usa un percentile basso (es. 15%) per stimare il livello di fondo da Y
            eps = torch.quantile(noise_image.float(), 0.15).item() 
            print(f"Epsilon value for 3d is {eps}")
        else:
            eps = eps_f
            print(f"Epsilon value for 2d is {eps}")

    wh_true = None
    if not is_realdata:
        ground_truth = parameters["ground_truth"]
        # Uso .view(-1, 1, 1, 1) così si adatta automaticamente a Nz
        clean_image = physics(ground_truth) + back_vec.view(-1, 1, 1, 1)
        clean_image_proc = clean_image.sum(1).unsqueeze(1) if is_3d else clean_image
        
        print(f"Noise image max: {noise_image.max().item()}")

        if mask_type == "masked":
            Z_true = standardize_unbiased_masked(noise_image, clean_image_proc)
        elif mask_type == "masked_eps":
            Z_true = standardize_unbiased_masked_eps(noise_image, clean_image_proc, eps)
        elif mask_type == "whole":
            Z_true = standardize(noise_image, clean_image_proc)
        else:
            raise ValueError(f"Metodo di masking non riconosciuto: {mask_type}")
        wh_mode = "highpass" if is_3d else "standard"
        wh_true, M_eff = whiteness_measure(Z_true, mode=wh_mode, cutoff=0.60)
        

    # --- 3. RICERCA SULLA GRIGLIA SIGMA ---
    best_result_dict = None
    sigma_best = None

    for i, sigma in enumerate(tqdm(sigma_values_grid, desc="Searching sigma grid (RWP)")):
        print(f"\n--- Testing sigma parameter = {sigma.item():.6f} ---")
                
        parameters["sigma"] = sigma
        
        # Esecuzione algoritmo PnP
        x_result, funct, iter_drunet, norm2_drunet, psnr_vec, ssim_vec = pnp_ism(noise_image, dataset['back_vec'], parameters, device)
      
        lambda_d = physics(x_result) + back_vec.view(-1, 1, 1, 1)
        if is_3d:
            lambda_d = lambda_d.sum(1).unsqueeze(1)
        
        # Calcolo di Z per il risultato corrente
        if mask_type == "masked":
            Z = standardize_unbiased_masked(noise_image, lambda_d) 
            Z_mean = torch.mean(Z)
            # Z = Z - Z_mean 
        elif mask_type == "masked_eps":
            Z = standardize_unbiased_masked_eps(noise_image, lambda_d, eps)
        elif mask_type == "whole":
            Z = standardize(noise_image, lambda_d)
        else:
            raise ValueError(f"Metodo di masking non riconosciuto: {mask_type}")
            
        # Metriche Whiteness
        wh_mode = "highpass" if is_3d else "standard"
        wh, M_eff = whiteness_measure(Z, mode=wh_mode, cutoff=cutoff)
        W_sum[i] = M_eff * wh
        
        # Calcolo Metriche PSNR/SSIM (solo per dati simulati)
        if not is_realdata:
            psnr_vecs[i] = psnr_vec[-1].item()
            ssim_vecs[i] = ssim_vec[-1].item()
            print(f"WP = {W_sum[i]:.4e} | PSNR = {psnr_vecs[i]:.2f} | SSIM = {ssim_vecs[i]:.4f}")
        else:
            print(f"WP = {W_sum[i]:.4e}")
        
        # Aggiornamento del minimo per WP
        if W_sum[i] < min_distance:
            # Salviamo i risultati in un dizionario per compatibilità con la funzione di plot
            best_result_dict = {
                "x_result": x_result,
                "funct": funct,
                "iter": iter_drunet,
                "norm2": norm2_drunet,
                "psnr_vec":psnr_vec,
                "ssim_vec":ssim_vec
            }
            sigma_best = sigma
            min_distance = W_sum[i]

    return W_sum, psnr_vecs, ssim_vecs, sigma_best, best_result_dict, wh_true

def compute_whiteness(x_curr, noise_image,  physics, back_vec, is_3d, mask_type = 'masked'):
        
    lambda_d = physics(x_curr) + back_vec.view(-1, 1, 1, 1)
    if is_3d :
        lambda_d = lambda_d.sum(1).unsqueeze(1)
    
    # Calcolo di Z per il risultato corrente
    if mask_type == "masked":
        Z = standardize_unbiased_masked(noise_image, lambda_d) 
        Z_mean = torch.mean(Z)
    elif mask_type == "masked_eps":
        # x1 = x_curr[:,0:1] 
        # # x1_masked = x1[x1 != 0]
        # eps = x1.mean() if is_3d else eps_f
        Z = standardize_unbiased_masked_eps(noise_image, lambda_d, eps = 1-5)
        Z_mean = torch.mean(Z)
    elif mask_type == "whole":
        Z = standardize(noise_image, lambda_d)
        Z_mean = torch.mean(Z)
    else:
        raise ValueError(f"Metodo di masking non riconosciuto: {mask_type}")
    
    # Metriche Whiteness
    wh, M_eff = whiteness_measure(Z, mode="highpass", cutoff=0.10)
    W_sum = M_eff * wh
    
    return W_sum, Z_mean

# def RWP_Adam_1Step(dataset, parameters, hparams, optim=Pgd_Backtracking,
#                    algorithm="pgd", mask_type="masked", eps=1, max_outer_iter=100, lrate = 5e-3, stepsize = 20, gamma = 0.5):
    
#     noise_image = dataset["noise_image"]
#     back_vec = dataset["back_vec"]
#     is_3d = hparams['IS_3D']
#     is_realdata = hparams['IS_REAL']
#     device = noise_image.device
#     M = noise_image.numel() 
    
#     physics = parameters["physics"]
#     SolverClass = optim

#     # 1. PARAMETRO DA OTTIMIZZARE
#     mu_raw = torch.tensor([0.001], dtype=torch.float32, device=device, requires_grad=True)
#     optimizer = torch.optim.Adam([mu_raw], lr=lrate)
#     scheduler = StepLR(optimizer, step_size=stepsize, gamma= gamma)

#     best_rwp = float('inf')
#     best_mu = None
    
#     RWP_vec = torch.empty(max_outer_iter, device=device)  
    
#     # Inizializzazione del punto di partenza per il "warm start"
#     x_0 = parameters["x_init"].clone()
    

#     print("\n--- Inizio Ottimizzazione di mu tramite Adam (1-Step Unrolling) ---")

#     for outer_idx in range(max_outer_iter):
#         optimizer.zero_grad()
#         mu = 0.1 * torch.sigmoid(mu_raw)
                
#         # ==========================================================
#         # FASE 1: CONVERGENZA SENZA GRADIENTE (Risparmio Memoria)
#         # ==========================================================
#         with torch.no_grad():
#             # Impostiamo lam disconnesso dal grafo per questa fase
#             parameters['lam'] = mu.item() 
#             parameters['x_init'] = x_0
            
#             if outer_idx < 100:
#                 parameters['max_iter'] = 1000  # Numero di iterazioni per arrivare a convergenza
#             else:
#                 parameters['max_iter'] = 10000
#             solver_no_grad = SolverClass(parameters, algorithm=algorithm, is_3d=is_3d, is_realdata=is_realdata)
#             results_no_grad = solver_no_grad.solve(y=noise_image)
            
#             # x a convergenza (staccato dal grafo)
#             x_converged = results_no_grad['x_curr'].detach()
            
#         # ==========================================================
#         # FASE 2: 1 SINGOLA ITERAZIONE CON GRADIENTE
#         # ==========================================================
#         # Qui passiamo il tensore `mu` (che ha requires_grad=True)
#         parameters['lam'] = mu 
#         parameters['x_init'] = x_converged # Partiamo esattamente dal punto di convergenza
#         parameters['max_iter'] = 1         # SOLO 1 ITERAZIONE per tenere traccia del grafo
        
#         solver_grad = SolverClass(parameters, algorithm=algorithm, is_3d=is_3d, is_realdata=is_realdata)
        
#         # Questo solve farà 1 solo step, costruendo un grafo leggerissimo
#         results_grad = solver_grad.solve(y=noise_image)
#         x_final = results_grad['x_result']
        
#         # ==========================================================
#         # FASE 3: CALCOLO LOSS RWP E BACKPROPAGATION
#         # ==========================================================
#         lambda_d = physics(x_final) + back_vec.view(-1, 1, 1, 1)
#         if is_3d:
#             lambda_d = lambda_d.sum(1).unsqueeze(1)
        
#         if mask_type == "masked":
#             Z = standardize_unbiased_masked(noise_image, lambda_d)
#         elif mask_type == "masked_eps":
#             Z = standardize_unbiased_masked_eps(noise_image, lambda_d, eps)
#         elif mask_type == "whole":
#             Z = standardize(noise_image, lambda_d)
        
#         # Calcolo RWP e moltiplicazione per M
#         loss_rwp = whiteness_measure(Z) * M
        
#         # Backpropagation attraverso l'unica iterazione
#         loss_rwp.backward()
#         optimizer.step()
#         scheduler.step()
        
#         if loss_rwp.item() < best_rwp:
#             best_rwp = loss_rwp.item()
#             best_mu = mu.item()
#             best_results = results_no_grad

#         if outer_idx % 5 == 0:
#             print(f"Iter {outer_idx:03d} | mu: {mu.item():.6f} | RWP: {loss_rwp.item():.4f}")
            
#         RWP_vec[outer_idx] = loss_rwp.item()

#     print(f"\nOttimizzazione completata. Miglior mu trovato: {best_mu:.6f} con RWP = {best_rwp:.4f}")
#     return best_mu, best_rwp, RWP_vec, best_results






# def find_knee_derivative(mu_values, W_sum): 
#     # Calcola il gradiente (derivata prima) 
#     dy = np.gradient(W_sum) 
#     # Calcola la derivata seconda 
#     ddy = np.gradient(dy) 
#     # Il punto di massimo cambiamento di pendenza 
#     best_idx = np.argmax(ddy) 
    
#     return mu_values[best_idx]


# def find_knee_threshold(mu_values, W_sum, threshold=1.10):
#     # Media dei primi punti (assumiamo siano nel plateau)
#     plateau_val = np.mean(W_sum[:3])
    
#     # Trova il primo indice che supera il plateau del 10%
#     for i, val in enumerate(W_sum):
#         if val > plateau_val * threshold:
#             # Ritorniamo il punto attuale o quello precedente per sicurezza
#             return mu_values[max(0, i-1)] 
            
#     return mu_values[0]

# def select_best_mu(mu_grid, W_sum):
#     """
#     Sceglie il mu ottimale basandosi sulla forma della curva RWP:
#     - Se esiste un minimo locale chiaro (una valle), lo seleziona.
#     - Se la curva è piatta (plateau) e poi sale, usa il metodo del gomito (Kneedle).
#     """
#     # Conversione in numpy (gestisce sia Liste che Tensor PyTorch)
#     mu_np = mu_grid.cpu().numpy() if hasattr(mu_grid, 'cpu') else np.array(mu_grid)
#     W_np = W_sum.cpu().numpy() if hasattr(W_sum, 'cpu') else np.array(W_sum)
    
#     idx_min = np.argmin(W_np)
    
#     # Verifichiamo se il minimo è "stabile" o è solo l'inizio di un plateau.
#     # Un minimo è reale se non è il primo punto e se dopo di esso c'è una risalita.
#     is_true_minimum = False
#     if 0 < idx_min < len(W_np) - 1:
#         # Se il valore finale è significativamente più alto del minimo (>5%),
#         # consideriamo la curva come dotata di una valle reale.
#         if W_np[-1] > W_np[idx_min] * 3:
#             is_true_minimum = True

#     if is_true_minimum:
#         # CASO 1: Esiste una valle chiara
#         return mu_np[idx_min]
#     else:
#         # CASO 2: Plateau o curva crescente (Metodo del Gomito / Kneedle)
#         # Operiamo in scala logaritmica per mu
#         log_mu = np.log10(mu_np + 1e-10)
        
#         # Normalizzazione tra 0 e 1 per rendere le distanze geometriche coerenti
#         w_norm = (W_np - W_np.min()) / (W_np.max() - W_np.min() + 1e-12)
#         mu_norm = (log_mu - log_mu.min()) / (log_mu.max() - log_mu.min() + 1e-12)
        
#         # Vettore che unisce il primo e l'ultimo punto della curva
#         p1 = np.array([mu_norm[0], w_norm[0]])
#         p2 = np.array([mu_norm[-1], w_norm[-1]])
#         line_vec = p2 - p1
#         line_norm = np.linalg.norm(line_vec)
        
#         # Calcolo della distanza di ogni punto dalla retta p1-p2
#         distances = []
#         for i in range(len(mu_norm)):
#             p3 = np.array([mu_norm[i], w_norm[i]])
#             # Distanza punto-retta tramite prodotto vettoriale 2D
#             d = np.abs(np.cross(line_vec, p1 - p3)) / line_norm
#             distances.append(d)
            
#         return mu_np[np.argmax(distances)]
    
    
    
    
# def select_best_mu_min(mu_grid, W_sum):
#     """
#     Seleziona il mu ottimale bilanciando il minimo globale e il punto di gomito.
#     - Se il minimo è a sinistra ma la pendenza per arrivarci è trascurabile, sceglie il gomito.
#     - Se il minimo è una valle profonda, lo segue.
#     """
#     mu_np = mu_grid.cpu().numpy() if hasattr(mu_grid, 'cpu') else np.array(mu_grid)
#     W_np = W_sum.cpu().numpy() if hasattr(W_sum, 'cpu') else np.array(W_sum)
    
#     # 1. Prepariamo le coordinate normalizzate (scala logaritmica per mu)
#     log_mu = np.log10(mu_np + 1e-10)
#     w_norm = (W_np - W_np.min()) / (W_np.max() - W_np.min() + 1e-12)
#     mu_norm = (log_mu - log_mu.min()) / (log_mu.max() - log_mu.min() + 1e-12)
    
#     # 2. Calcolo del vettore "Metodo del Gomito" (Distanza dalla retta corda)
#     p1 = np.array([mu_norm[0], w_norm[0]])
#     p2 = np.array([mu_norm[-1], w_norm[-1]])
#     line_vec = p2 - p1
#     line_norm = np.linalg.norm(line_vec)
    
#     # Calcoliamo la distanza perpendicolare per ogni punto
#     distances = []
#     for i in range(len(mu_norm)):
#         p3 = np.array([mu_norm[i], w_norm[i]])
#         d = np.abs(np.cross(line_vec, p1 - p3)) / line_norm
#         distances.append(d)
    
#     # 3. IDENTIFICAZIONE DEL PUNTO
#     # In una curva a "J" (plateau poi salita), il punto a distanza massima 
#     # dalla corda è esattamente il gomito dove la salita inizia.
#     # In una curva a "U" (minimo chiaro), il punto a distanza massima 
#     # coincide quasi perfettamente con il minimo locale.
    
#     best_idx = np.argmax(distances)
    
#     # Se il punto a distanza massima è troppo vicino all'inizio (estrema sinistra),
#     # significa che non c'è una salita reale, allora restituiamo il primo punto.
#     # Altrimenti, il punto a distanza massima è il miglior compromesso "gomito/valle".
    
#     return mu_np[best_idx]
