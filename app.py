from flask import Flask, render_template, request, session
import numpy as np
import matplotlib
import json
from sklearn.linear_model import LinearRegression
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt

app = Flask(__name__)
app.secret_key = "your_secret_key_here"  # Replace with your own secret key

# Data file path for storing generated data
DATA_FILE = "generated_data.json"


def generate_data(N, mu, beta0, beta1, sigma2, S):
    # Generate data and initial plots
    X = np.random.uniform(0, 1, N)
    error = np.random.normal(mu, np.sqrt(sigma2), N)
    Y = beta0 + beta1 * X + error

    model = LinearRegression()
    model.fit(X.reshape(-1, 1), Y)
    slope = model.coef_[0]
    intercept = model.intercept_

    # Save scatter plot with regression line
    plot1_path = "static/plot1.png"
    plt.figure()
    plt.scatter(X, Y, color='blue', label='Data Points')
    plt.plot(X, model.predict(X.reshape(-1, 1)), color='red', label='Regression Line')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.title('Scatter Plot with Regression Line')
    plt.savefig(plot1_path)
    plt.close()

    # Run simulations and save slopes and intercepts histograms
    slopes, intercepts = [], []
    for _ in range(S):
        X_sim = np.random.uniform(0, 1, N)
        error_sim = np.random.normal(mu, np.sqrt(sigma2), N)
        Y_sim = beta0 + beta1 * X_sim + error_sim

        sim_model = LinearRegression()
        sim_model.fit(X_sim.reshape(-1, 1), Y_sim)
        slopes.append(sim_model.coef_[0])
        intercepts.append(sim_model.intercept_)

    plot2_path = "static/plot2.png"
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.hist(slopes, bins=30, color='blue', alpha=0.7)
    plt.axvline(slope, color='red', linestyle='dashed', linewidth=2, label='Observed Slope')
    plt.title('Histogram of Slopes')
    plt.xlabel('Slope')
    plt.ylabel('Frequency')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.hist(intercepts, bins=30, color='green', alpha=0.7)
    plt.axvline(intercept, color='red', linestyle='dashed', linewidth=2, label='Observed Intercept')
    plt.title('Histogram of Intercepts')
    plt.xlabel('Intercept')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot2_path)
    plt.close()

    return X, Y, slope, intercept, plot1_path, plot2_path, slopes, intercepts


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        return generate()
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    # Get user input from the form
    N = int(request.form["N"])
    mu = float(request.form["mu"])
    sigma2 = float(request.form["sigma2"])
    beta0 = float(request.form["beta0"])
    beta1 = float(request.form["beta1"])
    S = int(request.form["S"])

    # Generate data and initial plots
    X, Y, slope, intercept, plot1, plot2, slopes, intercepts = generate_data(N, mu, beta0, beta1, sigma2, S)

    # Store essential data in JSON file
    data = {
        "N": N, "mu": mu, "sigma2": sigma2, "beta0": beta0, "beta1": beta1, "S": S,
        "slope": slope, "intercept": intercept, "slopes": slopes, "intercepts": intercepts
    }
    with open(DATA_FILE, "w") as f:
        json.dump(data, f)

    return render_template("index.html", plot1=plot1, plot2=plot2)


@app.route("/hypothesis_test", methods=["POST"])
def hypothesis_test():
    # Load data from JSON file
    with open(DATA_FILE, "r") as f:
        data = json.load(f)

    N = data["N"]
    S = data["S"]
    mu = data["mu"]
    sigma2 = data["sigma2"]
    beta0 = data["beta0"]
    beta1 = data["beta1"]
    slope = data["slope"]
    intercept = data["intercept"]
    slopes = data["slopes"]
    intercepts = data["intercepts"]

    parameter = request.form.get("parameter_ht")
    test_type = request.form.get("test_type_ht")

    # Use the slopes or intercepts from the simulations
    if parameter == "slope":
        simulated_stats = np.array(slopes)
        observed_stat = slope
        hypothesized_value = beta1
    else:
        simulated_stats = np.array(intercepts)
        observed_stat = intercept
        hypothesized_value = beta0

    # Calculate p-value based on test type
    if test_type == '>':
        p_value = np.sum(simulated_stats >= observed_stat) / S
    elif test_type == '<':
        p_value = np.sum(simulated_stats <= observed_stat) / S
    elif test_type == '!=':
        observed_diff = np.abs(observed_stat - hypothesized_value)
        simulated_diffs = np.abs(simulated_stats - hypothesized_value)
        p_value = np.sum(simulated_diffs >= observed_diff) / S
    else:
        p_value = None

    fun_message = "Wow! You've encountered a rare event with a tiny p-value!" if p_value <= 0.0001 else None

    # Plot histogram of simulated statistics
    plot3_path = "static/plot3.png"
    plt.figure()
    plt.hist(simulated_stats, bins=30, color='blue', alpha=0.7)
    plt.axvline(observed_stat, color='red', linestyle='dashed', linewidth=2, label='Observed Statistic')
    plt.axvline(hypothesized_value, color='green', linestyle='dashed', linewidth=2, label='Hypothesized Value')
    plt.xlabel('Simulated Statistics')
    plt.ylabel('Frequency')
    plt.title('Histogram of Simulated Statistics')
    plt.legend()
    plt.savefig(plot3_path)
    plt.close()

    return render_template(
        "index.html",
        plot1="static/plot1.png",
        plot2="static/plot2.png",
        plot3=plot3_path,
        p_value=p_value,
        fun_message=fun_message,
        parameter=parameter,
        observed_stat=observed_stat,
        hypothesized_value=hypothesized_value
    )


@app.route("/confidence_interval", methods=["POST"])
def confidence_interval():
    print("Session in confidence_interval:", session)

    # 尝试读取并验证数据
    try:
        with open(DATA_FILE, "r") as f:
            data = json.load(f)

        N = data["N"]
        mu = data["mu"]
        sigma2 = data["sigma2"]
        beta0 = data["beta0"]
        beta1 = data["beta1"]
        S = data["S"]
        slope = data["slope"]
        intercept = data["intercept"]
        slopes = data["slopes"]
        intercepts = data["intercepts"]
    except Exception as e:
        print("Error loading data from file:", e)
        return "Error loading data.", 500

    # 获取参数
    parameter = request.form.get("parameter_ci")
    confidence_level = float(request.form.get("confidence_level_ci"))

    # 根据参数类型选择估计值和真实值
    if parameter == "slope":
        estimates = np.array(slopes)
        observed_stat = slope
        true_param = beta1
    else:
        estimates = np.array(intercepts)
        observed_stat = intercept
        true_param = beta0

    # 计算均值和标准差
    mean_estimate = np.mean(estimates)
    std_estimate = np.std(estimates, ddof=1)

    # 计算置信区间
    alpha = 1 - (confidence_level / 100)
    t_crit = stats.t.ppf(1 - alpha / 2, df=S-1)
    margin_of_error = t_crit * (std_estimate / np.sqrt(S))

    ci_lower = mean_estimate - margin_of_error
    ci_upper = mean_estimate + margin_of_error
    includes_true = ci_lower <= true_param <= ci_upper

    # 绘制置信区间图
    plot4_path = "static/plot4.png"
    plt.figure(figsize=(10, 6))
    plt.scatter(range(S), estimates, color='gray', alpha=0.5, label='Simulated Estimates')
    plt.hlines([ci_lower, ci_upper], 0, S-1, colors='orange', linestyles='dashed', label='Confidence Interval')
    plt.axhline(true_param, color='green', linestyle='solid', label='True Parameter')
    plt.title(f'Confidence Interval for {parameter.capitalize()}')
    plt.legend()
    plt.savefig(plot4_path)
    plt.close()

    # 返回渲染后的模板，确保 `parameter` 在上下文中
    return render_template(
        "index.html",
        plot1="static/plot1.png",
        plot2="static/plot2.png",
        plot4=plot4_path,
        parameter=parameter,
        confidence_level=confidence_level,
        mean_estimate=mean_estimate,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        includes_true=includes_true,
        observed_stat=observed_stat,
        N=N,
        mu=mu,
        sigma2=sigma2,
        beta0=beta0,
        beta1=beta1,
        S=S,
    )



if __name__ == "__main__":
    app.run(debug=True)
