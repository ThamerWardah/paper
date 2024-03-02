from flask import Flask,send_file, jsonify, request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import det
from scipy.stats import chi2, multivariate_normal
import os
import io
from sklearn.covariance import MinCovDet
app = Flask(__name__)

@app.route("/mypage/<int:n>/<int:p>/<int:q>")
def my_page(n,p,q):
    # Control Section
    n = n
    p = p
    q = q
    mm_method = False  # else MCD method

    # New Control Section
    n1 = int(n * 0.8)
    n2 = n - n1
    VH = q
    VE = n - q - 1

    # Other variables
    v = 5
    c = 0.0625
    r = 0
    s = np.array([1, 1, 1, 1, 1, 1])
    testing = np.array([s])
    rsI = testing[0, :p]
    mI = np.array([])

    for i in range(p):
        for j in range(p):
            rj = 1 if i == j else r
            mI = np.append(mI, rj * rsI[i] * rsI[j])

    II = mI.reshape(p, p)

    b1 = 2
    b2 = 1.25
    d0 = np.sqrt(p) + b1 / np.sqrt(2)

    # Initializing arrays
    chi_in_H0_normal_Hampel = []
    chi_in_H0_normal_Huber = []
    chi_in_H0_outlier_Hampel = []
    chi_in_H0_outlier_Huber = []

    Huber_normal_H0 = []
    Huber_outlier_H1 = []
    Huber_normal_H1 = []

    Hampel_normal_H0 = []
    Hampel_outlier_H1 = []
    Hampel_normal_H1 = []

    classic_normal_H0 = []
    classic_outlier_H1 = []
    classic_normal_H1 = []

    p_value_robust_Huber_outlier = []
    p_value_classic_outlier = []
    p_value_robust_Hampel_outlier = []

    p_value_robust_Huber_normal = []
    p_value_classic_normal = []
    p_value_robust_Hampel_normal = []

    # Iteration
    iteration = 200
    iter = iteration / 5

    for outlier in [True, False]:
        for alpha_Power in [True, False]:
            for classic in [True,False]:
                if not classic:
                    for Hampel_cut in [True, False]:
                        np.random.seed(n*p*q)
                        
                        b = np.random.uniform(0, 0.6, size=(p, q+1))
                        if alpha_Power:
                            B = np.zeros_like(b)
                        else:
                            B = b
                        
                        sumwi = 0
                        for i in range(iteration):
                            # Generate xi~iid N(0,1)
                            Mu = np.zeros(q)
                            Sigma = np.eye(q)
                            
                            X1 = np.random.multivariate_normal(Mu, Sigma, n1)
                            X1 = np.hstack((np.ones((n1, 1)), X1))
                            
                            X2 = np.random.multivariate_normal(Mu, Sigma, n2)
                            X2 = np.hstack((np.ones((n2, 1)), X2))
                            

                            mu1 = np.zeros(p)
                            mu2 = v * np.sqrt(chi2.ppf(0.99, p)) * np.ones(p)
                            
                            e1 = np.random.multivariate_normal(mu1, II, n1)
                            if outlier:
                                e2 = np.random.multivariate_normal(mu2, c*II, n2)
                            else:
                                e2 = np.random.multivariate_normal(mu1, II, n2)
                            
                            Y1 = np.dot(X1, B.T) + e1
                            Y2 = np.dot(X2, B.T) + e2
                            
                            X = np.vstack((X1, X2))
                            Y = np.vstack((Y1, Y2))
                            
                            if mm_method:
                                one = rrcov.CovMMest(Y)
                            else:
                                mcd = MinCovDet().fit(Y)
                                mj = mcd.location_
                                Sj = mcd.covariance_
                            
                            work = 0
                            while work < 1:
                                d = np.zeros(n)
                                for i in range(n):
                                    d[i] = np.sqrt(np.dot(Y[i,:] - mj, np.linalg.solve(Sj, (Y[i,:] - mj))))
                                wi = np.zeros(n)
                                if not Hampel_cut:
                                    for di in range(n):
                                        if d[di] > chi2.ppf(0.95, p):
                                            wi[di] = 0
                                        else:
                                            wi[di] = 1
                                else:
                                    for di in range(n):
                                        if d[di] > d0:
                                            wi[di] = (d0 * np.exp(-0.5 * ((d[di] - d0) / b2) ** 2)) / d[di]
                                        else:
                                            wi[di] = 1
                                sumwi += np.sum(wi)
                                mmj2 = np.zeros(p)
                                for ii in range(n):
                                    mmj2 += wi[ii] * Y[ii,:]
                                mj2 = mmj2 / np.sum(wi)
                                SSj2 = np.zeros((p, p))
                                for jj in range(n):
                                    SSj2 += wi[jj] * np.outer(Y[jj,:] - mj2, Y[jj,:] - mj2)
                                Sj2 = SSj2 / (np.sum(wi) - 1)
                                if (np.sum(np.diag(Sj2 / p)) ** p / np.linalg.det(Sj2)) <= (np.sum(np.diag(Sj / p)) ** p / np.linalg.det(Sj)):
                                    work = 1
                                Sjj = Sj
                                mjj = mj
                                Sj = Sj2
                                mj = mj2
                            
                            W = np.diag(wi)
                            Jw = np.outer(wi, wi)
                            
                            SE_w = W - np.dot(W, np.dot(X, np.dot(np.linalg.inv(np.dot(X.T, np.dot(W, X))), np.dot(X.T, W))))
                            df_VE = np.trace(SE_w)
                            
                            SR_w = np.dot(W, np.dot(X, np.dot(np.linalg.inv(np.dot(X.T, np.dot(W, X))), np.dot(X.T, W)))) - (Jw / np.sum(wi))
                            df_VH = np.trace(SR_w)
                            
                            SSE_w = np.dot(Y.T, np.dot(SE_w, Y))
                            SSR_w = np.dot(Y.T, np.dot(SR_w, Y))
                            
                            wilks = np.linalg.det(SSE_w) / np.linalg.det(SSE_w + SSR_w)

                            chi_w = -((df_VE) - 0.5 * (p - (df_VH) + 1)) * np.log(wilks)
                            fdw = p * df_VH
                            
                            if not Hampel_cut:
                                if alpha_Power and not outlier:
                                    Huber_normal_H0.append(chi_w)
                                    chi_in_H0_normal_Huber.append(chi_w)
                                if alpha_Power and outlier:
                                    chi_in_H0_outlier_Huber.append(chi_w)
                                if not alpha_Power and outlier:
                                    Huber_outlier_H1.append(chi_w)
                                if not alpha_Power and not outlier:
                                    Huber_normal_H1.append(chi_w)
                            
                            if Hampel_cut:
                                if alpha_Power and not outlier:
                                    Hampel_normal_H0.append(chi_w)
                                    chi_in_H0_normal_Hampel.append(chi_w)
                                if alpha_Power and outlier:
                                    chi_in_H0_outlier_Hampel.append(chi_w)
                                if not alpha_Power and outlier:
                                    Hampel_outlier_H1.append(chi_w)
                                if not alpha_Power and not outlier:
                                    Hampel_normal_H1.append(chi_w)
                            
                            if outlier and alpha_Power:
                                if not Hampel_cut:
                                    p_value_robust_Huber_outlier.append(1 - chi2.cdf(chi_w, fdw))
                                else:
                                    p_value_robust_Hampel_outlier.append(1 - chi2.cdf(chi_w, fdw))
                            if not outlier and alpha_Power:
                                if not Hampel_cut:
                                    p_value_robust_Huber_normal.append(1 - chi2.cdf(chi_w, fdw))
                                else:
                                    p_value_robust_Hampel_normal.append(1 - chi2.cdf(chi_w, fdw))


                else:
                    np.random.seed(n * p * q)
                    b = np.random.uniform(0, 0.6, size=(p, q + 1))
                    if alpha_Power:
                        B = b * 0
                    else:
                        B = b

                    for jclassic in range(iteration):
                        X1 = np.random.multivariate_normal(np.zeros(q), np.eye(q), size=n1)
                        X1 = np.hstack((np.ones((n1, 1)), X1))

                        X2 = np.random.multivariate_normal(np.zeros(q), np.eye(q), size=n2)
                        X2 = np.hstack((np.ones((n2, 1)), X2))

                        mu1 = np.zeros(p)
                        mu2 = v * np.sqrt(chi2.ppf(0.99, p)) * np.ones(p)

                        e1 = np.random.multivariate_normal(mu1, II, size=n1)
                        if outlier:
                            e2 = np.random.multivariate_normal(mu2, c * II, size=n2)
                        else:
                            e2 = np.random.multivariate_normal(mu1, II, size=n2)
                        
                        Y1 = np.dot(X1, B.T) + e1
                        Y2 = np.dot(X2, B.T) + e2

                        X = np.vstack((X1, X2))
                        Y = np.vstack((Y1, Y2))

                        B0 = np.linalg.inv(X.T @ X) @ X.T @ Y

                        Yb = np.array([np.mean(Y[:, k]) for k in range(p)])

                        H = B0.T @ X.T @ Y - n * np.outer(Yb, Yb.T)
                        E = Y.T @ Y - B0.T @ X.T @ Y

                        wik = det(E) / (det(E + H))

                        qi = -(VE - 0.5 * (p - VH + 1)) * np.log(wik)

                        if alpha_Power and not outlier:
                            classic_normal_H0.append(qi)

                        if not alpha_Power and outlier:
                            classic_outlier_H1.append(qi)

                        if not alpha_Power and not outlier:
                            classic_normal_H1.append(qi)

                        if outlier and alpha_Power:
                            p_value_classic_outlier.append(1 - chi2.cdf(qi, p * VH))

                        if not outlier and alpha_Power:
                            p_value_classic_normal.append(1 - chi2.cdf(qi, p * VH))

    # Sort arrays in descending order
    Huber_normal_H0 = np.sort(Huber_normal_H0)[::-1]
    Hampel_normal_H0 = np.sort(Hampel_normal_H0)[::-1]
    classic_normal_H0 = np.sort(classic_normal_H0)[::-1]


    outlier_power_size = np.array([])
    normal_power_size = np.array([])

    for just_200_first in range(1, 201):
        sum_outlier = 0
        for just_1000_second in range(1, iteration + 1):
            if Huber_outlier_H1[just_1000_second - 1] > Huber_normal_H0[just_200_first - 1]:
                sum_outlier += 1
        outlier_power_size = np.append(outlier_power_size, sum_outlier)

        sum_normal = 0
        for just_1000_second in range(1, iteration + 1):
            if Huber_normal_H1[just_1000_second - 1] > Huber_normal_H0[just_200_first - 1]:
                sum_normal += 1
        normal_power_size = np.append(normal_power_size, sum_normal)


    Hampel_outlier_power_size = np.array([])
    Hampel_normal_power_size = np.array([])

    for just_200_first in range(1, 201):
        Hampel_sum_outlier = 0
        for just_1000_second in range(1, iteration + 1):
            if Hampel_outlier_H1[just_1000_second - 1] > Hampel_normal_H0[just_200_first - 1]:
                Hampel_sum_outlier += 1
        Hampel_outlier_power_size = np.append(Hampel_outlier_power_size, Hampel_sum_outlier)

        Hampel_sum_normal = 0
        for just_1000_second in range(1, iteration + 1):
            if Hampel_normal_H1[just_1000_second - 1] > Hampel_normal_H0[just_200_first - 1]:
                Hampel_sum_normal += 1
        Hampel_normal_power_size = np.append(Hampel_normal_power_size, Hampel_sum_normal)


    classic_outlier_power_size = np.array([])
    classic_normal_power_size = np.array([])

    for just_200_first in range(1, 201):
        classic_sum_outlier = 0
        for just_1000_second in range(1, iteration + 1):
            if classic_outlier_H1[just_1000_second - 1] > classic_normal_H0[just_200_first - 1]:
                classic_sum_outlier += 1
        classic_outlier_power_size = np.append(classic_outlier_power_size, classic_sum_outlier)

        classic_sum_normal = 0
        for just_1000_second in range(1, iteration + 1):
            if classic_normal_H1[just_1000_second - 1] > classic_normal_H0[just_200_first - 1]:
                classic_sum_normal += 1
        classic_normal_power_size = np.append(classic_normal_power_size, classic_sum_normal)

    #draw the size power code 
    # Create x_draw
    x_draw = np.array([])
    for draw in range(1, 201):
        significant2 = draw * 0.001
        x_draw = np.append(x_draw, significant2)

    x_axis = x_draw.reshape(-1, 1)

    # Calculate power sizes
    power_sizes_normal = {
        "Robust_normal_Huber":normal_power_size/iteration,
        "Robust_normal_Hampel":Hampel_normal_power_size/iteration,
        "classic_normal":classic_normal_power_size/iteration
    }

    power_sizes_outliter = {
        "Robust_outlier_Huber":outlier_power_size/iteration,
        "Robust_outlier_Hampel":Hampel_outlier_power_size/iteration,
        "classic_outlier":classic_outlier_power_size/iteration
    }

    # Plotting
    custom_colors = {
        "Robust_outlier_Huber": 'black',
        "Robust_normal_Huber": 'black',
        "Robust_outlier_Hampel": 'green',
        "Robust_normal_Hampel": 'green',
        "classic_outlier": 'blue',
        "classic_normal": 'blue'
    }

    line_45 = np.arange(0, 0.2, 0.0025)

    plt.figure(figsize=(5, 5.5))

    for key, values in power_sizes_outliter.items():
        plt.plot(x_axis, values, label=key, color=custom_colors[key])

    plt.plot(line_45, line_45, color='gray', linestyle='--', linewidth=0.4)

    plt.title('r={} n={} p={} q={}'.format(r, n, p, q))
    plt.xlabel('Size')
    plt.ylabel('Power')
    plt.legend(loc='upper right', fontsize=14, title_fontsize=14)
    plt.grid(False)
    plt.ylim(0, 1)
    plt.gca().set_aspect(0.2, adjustable='box')
    img = io.BytesIO()
    plt.savefig(img,format='png')
    img.seek(0)
    plt.close()
    return send_file(img,mimetype="image/png")

if __name__ == "__main__":
    app.run(debug=True)