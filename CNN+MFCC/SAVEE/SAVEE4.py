import librosa
import librosa.display
import numpy as np
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from tensorflow.python.keras.utils.np_utils import to_categorical
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder, StandardScaler
from keras import datasets, layers, models
from keras.metrics import Precision, Recall
from keras.optimizers import Adam
from keras import regularizers
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Initialize feeling list
feeling_list = []

# Load file list
mylist = os.listdir('D:\Third year project\Dataset\SAVEE/')

# Define a new function to extract a single dominant emotion from filenames
def extract_emotion(filepath):
    filename = os.path.basename(filepath)
    if filename.startswith('n'):
        return 'neutral'
    elif filename.startswith('h'):
        return 'happy'
    elif filename.startswith('sa'):
        return 'sad'
    elif filename.startswith('a'):
        return 'angry'
    else:
        return None

# Extract emotions from each file
for item in mylist:
    emotion = extract_emotion(item)
    if emotion:
        feeling_list.append(emotion)

# Label DataFrame
labels = pd.DataFrame(feeling_list)

# Construct a DataFrame containing feature columns
df = pd.DataFrame(columns=['feature'])
bookmark = 0

for index, y in enumerate(mylist):
    if mylist[index][0:1] in ['n', 'h', 'a'] or mylist[index][0:2] in ['sa']:
        # Load
        X, sample_rate = librosa.load('D:\\Third year project\\Dataset\\SAVEE\\' + y, res_type='kaiser_fast', duration=2.5, sr=22050, offset=0.5)
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13)
        # Extract Mel Spectrogram and convert to decibels
        mel_spectrogram = librosa.feature.melspectrogram(y=X, sr=sample_rate, n_mels=128)
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        feature = np.hstack((
            np.mean(mfccs, axis=1),
            np.mean(mel_spectrogram_db, axis=1)
        ))
        # Add feature vector to DataFrame
        df.loc[bookmark] = [feature]
        bookmark += 1



# Integrate labels
df3 = pd.DataFrame(df['feature'].values.tolist())
newdf = pd.concat([df3, labels], axis=1)

# Rename columns
rnewdf = newdf.rename(index=str, columns={0: "label"})

# Mix order
rnewdf = shuffle(newdf)

features = rnewdf.iloc[:, :-1]  
labels = rnewdf.iloc[:, -1:]    

# Split
X_train, X_temp, y_train, y_temp = train_test_split(features, labels, test_size=2/15, random_state=42, stratify=labels)
X_validation, X_test, y_validation, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Label encoding for single-label classification
lb = LabelEncoder()
y_train = to_categorical(lb.fit_transform(y_train.values.ravel()))
y_validation = to_categorical(lb.transform(y_validation.values.ravel()))
y_test = to_categorical(lb.transform(y_test.values.ravel()))

#check
print(f"Training set size: {y_train.shape[0]}")
print(f"Validation set size: {y_validation.shape[0]}")
print(f"Test set size: {y_test.shape[0]}")

# Check
print(f"x_traincnn shape: {X_train.shape}")
print(f"x_validationcnn shape: {X_validation.shape}")
print(f"x_testcnn shape: {X_test.shape}")

# Expand dimensions for CNN input
x_traincnn = np.expand_dims(X_train.to_numpy(), axis=2)
x_validationcnn = np.expand_dims(X_validation.to_numpy(), axis=2)
x_testcnn = np.expand_dims(X_test.to_numpy(), axis=2)

#check
print(f"x_traincnn shape: {x_traincnn.shape}")
print(f"x_validationcnn shape: {x_validationcnn.shape}")
print(f"x_testcnn shape: {x_testcnn.shape}")

# Building a CNN Sequential Model with L2 regularization, dropout, and batch normalization
model = models.Sequential()
# First convolution layer with L2 regularization + activation + batch normalization
model.add(layers.Conv1D(256, 3, padding='same', input_shape=(X_train.shape[1], 1), kernel_regularizer=regularizers.l2(0.001)))
model.add(layers.Activation('relu'))
model.add(layers.BatchNormalization())
# Second convolution layer with L2 regularization + activation + batch normalization
model.add(layers.Conv1D(128, 3, padding='same', kernel_regularizer=regularizers.l2(0.001)))
model.add(layers.Activation('relu'))
model.add(layers.BatchNormalization())
# Dropout for overfitting prevention
model.add(layers.Dropout(0.5))
# Third convolution layer with L2 regularization + activation + batch normalization
model.add(layers.Conv1D(128, 3, padding='same', kernel_regularizer=regularizers.l2(0.001)))
model.add(layers.Activation('relu'))
model.add(layers.BatchNormalization())
# Fourth convolution layer with L2 regularization + activation + batch normalization
model.add(layers.Conv1D(128, 3, padding='same', kernel_regularizer=regularizers.l2(0.001)))
model.add(layers.Activation('relu'))
model.add(layers.BatchNormalization())
# Flatten layer
model.add(layers.Flatten())
# Dense layer (output layer) with L2 regularization
model.add(layers.Dense(len(lb.classes_), kernel_regularizer=regularizers.l2(0.001)))
model.add(layers.Activation('softmax'))

# Output model information
model.summary()

# Set learning rate
learning_rate = 0.00005

# Compile the model - This line is essential to compile before training
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=learning_rate), metrics=['accuracy', Precision(name='precision'), Recall(name='recall')])
# Train the model with a validation set
cnnhistory = model.fit(x_traincnn, y_train, batch_size=16, epochs=500, validation_data=(x_validationcnn, y_validation))

# Evaluate the model on the test set to get the final accuracy
score = model.evaluate(x_testcnn, y_test, verbose=0)
test_accuracy = score[1] * 100
print("Test Accuracy: %.2f%%" % test_accuracy)

# Print the last training and validation accuracies
last_training_accuracy = cnnhistory.history['accuracy'][-1] * 100
last_validation_accuracy = cnnhistory.history['val_accuracy'][-1] * 100
print("Final Training Accuracy: %.2f%%" % last_training_accuracy)
print("Final Validation Accuracy: %.2f%%" % last_validation_accuracy)

# Predict on the test set
preds = model.predict(x_testcnn, batch_size=32, verbose=1)
# Highest category
pred_labels = preds.argmax(axis=1)
# Predicted values and label
pred_labels = pred_labels.astype(int).flatten()
predictedvalues = lb.inverse_transform(pred_labels)
# Actual labels and values
actual_labels = y_test.argmax(axis=1).astype(int).flatten()
actualvalues = lb.inverse_transform(actual_labels)
# Combine
final_df = pd.DataFrame({'actualvalues': actualvalues, 'predictedvalues': predictedvalues})

# Output
print(final_df[160:190])

# Generate the confusion matrix
conf_matrix = confusion_matrix(actual_labels, pred_labels)

# Get the list of emotion labels (class names)
emotion_labels = lb.classes_  # Assuming `lb` is the LabelEncoder used earlier

# Calculate accuracy per emotion
print("Accuracy per emotion at test set:")
for i, emotion in enumerate(emotion_labels):
    true_positives = conf_matrix[i, i]
    total_instances = conf_matrix[i].sum()
    accuracy = true_positives / total_instances if total_instances > 0 else 0
    print(f"{emotion}: {accuracy * 100:.2f}%")
    
###Confusion matrix
def draw_confusion_matrix(label_true, label_pred, label_name, title="Confusion Matrix", pdf_save_path=None, dpi=100):
    """
    Draws a normalized confusion matrix with annotations.
    """
    # Generate confusion matrix with normalization
    cm = confusion_matrix(y_true=label_true, y_pred=label_pred, normalize='true')

    # Plot confusion matrix as an image
    plt.imshow(cm, cmap='Blues')
    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.yticks(range(len(label_name)), label_name)
    plt.xticks(range(len(label_name)), label_name, rotation=45)
    plt.tight_layout()
    plt.colorbar()

    # Annotate the matrix with values
    for i in range(len(label_name)):
        for j in range(len(label_name)):
            color = (1, 1, 1) if i == j else (0, 0, 0)  # White for diagonal, black for others
            value = float(format(cm[j, i], '.2f'))
            plt.text(i, j, value, verticalalignment='center', horizontalalignment='center', color=color)

    # Save to file if a path is provided
    if pdf_save_path is not None:
        plt.savefig(pdf_save_path, bbox_inches='tight', dpi=dpi)

    # Display the plot
    plt.show()

# Use this function after generating predictions
actual_labels = y_test.argmax(axis=1).astype(int).flatten()
predicted_labels = preds.argmax(axis=1).astype(int).flatten()

# Pass the parameters to the draw_confusion_matrix function
draw_confusion_matrix(
    label_true=actual_labels,
    label_pred=predicted_labels,
    label_name=emotion_labels,  # Use your list of emotion labels
    title="Confusion Matrix on Emotion Detection",
    pdf_save_path="Confusion_Matrix.png",  # Change or remove if saving isn't needed
    dpi=300
)


###Loss Curve
def plot_loss_curve(history, title="Loss Curve"):

    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
    
###Accuracy
def plot_precision_curve(history, title="Precision Curve"):

    plt.figure(figsize=(8, 6))
    plt.plot(history.history['precision'], label='Training Precision')
    plt.plot(history.history['val_precision'], label='Validation Precision')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Precision')
    plt.legend()
    plt.grid(True)
    plt.show()

###Recall
def plot_recall_curve(history, title="Recall Curve"):

    plt.figure(figsize=(8, 6))
    plt.plot(history.history['recall'], label='Training Recall')
    plt.plot(history.history['val_recall'], label='Validation Recall')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Recall')
    plt.legend()
    plt.grid(True)
    plt.show()

###Plot
plot_loss_curve(cnnhistory, title="Training and Validation Loss")
plot_precision_curve(cnnhistory, title="Training and Validation Precision")
plot_recall_curve(cnnhistory, title="Training and Validation Recall")


def generate_all_plots(history, label_true, label_pred, label_name, save_path=None, dpi=300):
    """
    一次性生成所有性能曲线（损失、精确率、召回率）和混淆矩阵。
    """
    # 创建子图布局
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))  # 增加图像大小

    # === 绘制 Loss 曲线 ===
    axes[0, 0].plot(history.history['loss'], label='Training Loss', linewidth=2)
    axes[0, 0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0, 0].set_title("Loss Curve", fontsize=14)
    axes[0, 0].set_xlabel("Epochs", fontsize=12)
    axes[0, 0].set_ylabel("Loss", fontsize=12)
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True)

    # === 绘制 Precision 曲线 ===
    axes[0, 1].plot(history.history['precision'], label='Training Precision', linewidth=2)
    axes[0, 1].plot(history.history['val_precision'], label='Validation Precision', linewidth=2)
    axes[0, 1].set_title("Precision Curve", fontsize=14)
    axes[0, 1].set_xlabel("Epochs", fontsize=12)
    axes[0, 1].set_ylabel("Precision", fontsize=12)
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True)

    # === 绘制 Recall 曲线 ===
    axes[1, 0].plot(history.history['recall'], label='Training Recall', linewidth=2)
    axes[1, 0].plot(history.history['val_recall'], label='Validation Recall', linewidth=2)
    axes[1, 0].set_title("Recall Curve", fontsize=14)
    axes[1, 0].set_xlabel("Epochs", fontsize=12)
    axes[1, 0].set_ylabel("Recall", fontsize=12)
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True)

    # === 绘制 Confusion Matrix ===
    cm = confusion_matrix(y_true=label_true, y_pred=label_pred, normalize='true')
    im = axes[1, 1].imshow(cm, cmap='Blues')
    axes[1, 1].set_title("Confusion Matrix", fontsize=14)
    axes[1, 1].set_xlabel("Predicted Label", fontsize=12)
    axes[1, 1].set_ylabel("True Label", fontsize=12)
    axes[1, 1].set_xticks(range(len(label_name)))
    axes[1, 1].set_yticks(range(len(label_name)))
    axes[1, 1].set_xticklabels(label_name, rotation=45, ha='right', fontsize=10)
    axes[1, 1].set_yticklabels(label_name, fontsize=10)

    # 在混淆矩阵上显示数值
    for i in range(len(label_name)):
        for j in range(len(label_name)):
            color = (1, 1, 1) if i == j else (0, 0, 0)  # 对角线白色字体
            value = float(format(cm[j, i], '.2f'))
            axes[1, 1].text(j, i, value, ha='center', va='center', color=color, fontsize=10)
    
    fig.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)  # 控制色条大小和间距

    # 调整全局标题
    fig.suptitle("Model Performance and Confusion Matrix", fontsize=18, y=0.98)

    # 调整子图间距
    fig.subplots_adjust(hspace=0.4, wspace=0.3)  # 增加子图垂直和水平间距

    # 保存或显示图像
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.show()

# 调用整合函数生成图像
generate_all_plots(
    history=cnnhistory,
    label_true=actual_labels,
    label_pred=predicted_labels,
    label_name=emotion_labels,
    save_path="Performance_Plots.png",  # 保存文件名，可改为 None 仅显示图像
    dpi=300
)

###Radar
# 假设已生成分类报告
report = classification_report(actual_labels, predicted_labels, target_names=emotion_labels, output_dict=True)

# 将分类报告转化为 DataFrame
metrics_df = pd.DataFrame(report).transpose()
metrics_df = metrics_df.loc[emotion_labels, ['precision', 'recall', 'f1-score']]  # 提取每类指标

# 绘制雷达图
def plot_multi_radar_chart(metrics_df, save_path=None):
    """
    绘制多个类别性能分布的雷达图，每个类别一条线。
    :param metrics_df: 包含 Precision、Recall、F1-Score 的 DataFrame
    :param save_path: 如果提供路径，则保存图片
    """
    labels = metrics_df.columns.tolist()  # 指标名称 ['precision', 'recall', 'f1-score']
    categories = metrics_df.index.tolist()  # 类别名称 ['angry', 'calm', 'fearful', ...]

    N = len(labels)  # 指标维度
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # 闭合雷达图

    # 初始化画布
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # 颜色样式
    colors = ['r', 'g', 'b', 'y', 'm', 'c']
    linestyles = ['-', '--', '-.', ':']
    
    # 绘制每个类别的指标
    for i, category in enumerate(categories):
        values = metrics_df.loc[category].values.tolist()
        values += values[:1]  # 闭合
        ax.plot(
            angles, 
            values, 
            color=colors[i % len(colors)],  # 循环使用颜色
            linestyle=linestyles[i % len(linestyles)],  # 循环使用线条样式
            linewidth=2, 
            label=category
        )
        ax.fill(angles, values, color=colors[i % len(colors)], alpha=0.25)

    # 设置特征标签
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_ylim(0, 1)  # 指标范围 0-1
    ax.set_title("Radar Chart for Model Performance", fontsize=16, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1), fontsize=10)
    ax.grid(True)

    # 保存或显示图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# 调用函数绘图
plot_multi_radar_chart(metrics_df, save_path="Multi_Radar_Chart.png")