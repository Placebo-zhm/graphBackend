import json
import numpy as np
import tensorflow as tf
from keras.layers import Bidirectional, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os

#加载预处理后的年度关键词数据
def load_processed_data(json_path="../data/jsondata/time_sequence.json"):
    with open(json_path, 'r', encoding='utf-8') as f:
        full_data = json.load(f)
    return full_data['data']

#获取指定关键词的完整时间序列
def get_keyword_sequence(full_data, keyword, min_year=2003, max_year=2024):
    # 获取有效年份范围
    all_years = sorted([int(y) for y in full_data.keys() if y.isdigit()])
    if not all_years:
        raise ValueError("未找到有效年份数据")

    # 生成连续年份序列
    min_year = max(min(all_years), min_year)
    max_year = min(max(all_years), max_year)
    year_range = list(range(min_year, max_year + 1))

    # 填充频次数据
    sequence = []
    for year in year_range:
        year_str = str(year)
        sequence.append(full_data.get(year_str, {}).get(keyword.lower(), 0))

    return np.array(sequence), year_range


# 修改后的预测函数（增加实际值验证）
def predict_future(keyword, n_years=5, validate_year=None, model_save_path=None):
    """
    Parameters:
    validate_year - 需要验证的年份（int，如2024）
    model_save_path - 模型保存路径（None表示不保存）
    """
    # 加载数据
    data = load_processed_data()
    sequence, years = get_keyword_sequence(data, keyword)

    print(sequence)
    print(years)
    print(len(years))

    # 如果有验证年份，分割数据集
    if validate_year and validate_year in years:
        val_index = years.index(validate_year)
        train_sequence = sequence[:val_index]
        val_true = sequence[val_index]

        # 调整预测年数为实际需要的间隔
        # n_years = validate_year - years[-1] if validate_year > years[-1] else 1
    else:
        train_sequence = sequence

    # 数据标准化
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(train_sequence.reshape(-1, 1))

    # 创建监督学习数据集
    def create_dataset(data, time_steps=3):
        X, y = [], []
        for i in range(len(data) - time_steps):
            X.append(data[i:(i + time_steps), 0])
            y.append(data[i + time_steps, 0])
        return np.array(X), np.array(y)

    time_steps = 3
    X, y = create_dataset(scaled_data, time_steps)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # 构建/加载模型
    if model_save_path and os.path.exists(model_save_path):
        model = tf.keras.models.load_model(model_save_path)
        print(f"Loaded existing model from {model_save_path}")
    else:
        # model = Sequential()
        # model.add(LSTM(50, activation='relu', input_shape=(time_steps, 1)))
        # model.add(Dense(1))
        # model.compile(optimizer='adam', loss='mse')
        model = Sequential([
            Bidirectional(LSTM(128, return_sequences=True),
                          input_shape=(time_steps, 1)),
            Dropout(0.3),
            LSTM(64, activation='tanh'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')

    # 训练模型
    if not (model_save_path and os.path.exists(model_save_path)):
        model.fit(X, y, epochs=200, verbose=0)
        if model_save_path:
            model.save(model_save_path)
            print(f"Model saved to {model_save_path}")

    # 进行预测
    current_batch = scaled_data[-time_steps:].reshape(1, time_steps, 1)
    predictions = []

    for _ in range(n_years):
        next_pred = model.predict(current_batch, verbose=0)[0]
        predictions.append(next_pred)
        current_batch = np.append(current_batch[:, 1:, :], [[next_pred]], axis=1)

    # 反标准化
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    predictions = np.round(predictions).astype(int)

    # 验证处理
    validation_result = {}
    if validate_year and validate_year in years:
        # 计算单步预测误差
        val_pred = predictions[0][0]
        validation_result = {
            'year': validate_year,
            'true_value': val_true,
            'predicted_value': val_pred,
            'mae': mean_absolute_error([val_true], [val_pred]),
            'mse': mean_squared_error([val_true], [val_pred])
        }
        # 更新历史数据用于后续预测
        sequence = np.append(sequence, val_true)
        years.append(validate_year)

    # 生成预测年份
    last_year = years[-1]
    prediction_years = list(range(last_year + 1, last_year + 1 + n_years))

    return {
        'history_years': years,
        'history_data': sequence.tolist(),
        'prediction_years': prediction_years,
        'predictions': predictions.flatten().tolist(),
        'validation': validation_result
    }


# 绘图函数
def plot_trend(result_dict, keyword):
    font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=12)

    plt.figure(figsize=(12, 6))

    # 历史数据
    plt.plot(result_dict['history_years'], result_dict['history_data'],
             'bo-', label='历史数据', markersize=8, linewidth=2)

    # 预测数据
    plt.plot(result_dict['prediction_years'], result_dict['predictions'],
             'r^--', label='预测数据', markersize=8, linewidth=2)

    # 验证点标注
    if result_dict['validation']:
        val = result_dict['validation']
        plt.scatter(val['year'], val['true_value'],
                    c='green', s=120, label='实际值', zorder=10)
        plt.scatter(val['year'], val['predicted_value'],
                    c='orange', s=120, label='验证预测值', zorder=10)

    # 数据标注
    for x, y in zip(result_dict['history_years'], result_dict['history_data']):
        plt.text(x, y, f'{y:.1f}', ha='center', va='bottom', fontproperties=font)

    for x, y in zip(result_dict['prediction_years'], result_dict['predictions']):
        plt.text(x, y, f'{y:.1f}', ha='center', va='bottom',
                 color='red', fontproperties=font)

    # 图表装饰
    title = f"'{keyword}' 年度趋势分析"
    if result_dict['validation']:
        title += f"\n验证结果 MAE={result_dict['validation']['mae']:.2f} MSE={result_dict['validation']['mse']:.2f}"
    plt.title(title, fontproperties=font)
    plt.xlabel("年份", fontproperties=font)
    plt.ylabel("出现频次", fontproperties=font)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(prop=font)

    # 调整坐标轴
    all_years = result_dict['history_years'] + result_dict['prediction_years']
    plt.xlim(min(all_years) - 1, max(all_years) + 1)

    plt.tight_layout()
    plt.show()


# 更新示例用法
if __name__ == "__main__":
    keyword = "visualization"
    #keyword = "visual analytics"
    model_path = f"{keyword}_lstm_model.h5"

    # 带验证的预测
    result = predict_future(
        keyword=keyword,
        n_years=10,
        #validate_year=2024,  # 需要验证的年份
        model_save_path=model_path  # 模型保存路径
    )

    # 打印验证结果
    if result['validation']:
        print("\n验证结果:")
        print(f"年份: {result['validation']['year']}")
        print(f"实际值: {result['validation']['true_value']:.1f}")
        print(f"预测值: {result['validation']['predicted_value']:.1f}")
        print(f"MAE: {result['validation']['mae']:.2f}")
        print(f"MSE: {result['validation']['mse']:.2f}")

    # 绘制趋势图
    plot_trend(result, keyword)

    # 打印预测结果
    print("\n未来预测:")
    print("年份:", result['prediction_years'])
    print("预测值:", [round(x, 1) for x in result['predictions']])