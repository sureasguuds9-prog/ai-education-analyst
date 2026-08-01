# 🎓 ИИ в образовании: помогает или вредит?

**Разведочный анализ связи GenAI-инструментов с успеваемостью, выгоранием и навыками студентов**

**Автор:** Yaroslav Zinchenko  
**Дата:** 2026-06-04  
**Датасет:** `ai_student_impact_dataset.csv` — 50 000 студентов, 16 признаков  
**Стек:** Python · pandas · numpy · scipy · matplotlib · seaborn

---

## Содержание

1. [Описание данных](#1-описание-данных)
2. [Загрузка и очистка](#2-загрузка-и-очистка)
3. [Общая картина](#3-общая-картина)
4. [Ценовой анализ: часы ИИ vs результат](#4-анализ-часы-ии-vs-результат)
5. [Связь с политикой вуза](#5-влияние-политики-вуза)
6. [Сравнение специальностей](#6-сравнение-специальностей)
7. [Сценарии использования ИИ](#7-сценарии-использования-ии)
8. [Статистические тесты](#8-статистические-тесты)
9. [Выводы и рекомендации](#9-выводы-и-рекомендации)

---

## 0. Настройка

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'figure.dpi': 120,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'font.size': 10,
})
sns.set_palette('Set2')
```

---

## 1. Описание данных

### Что содержит датасет

| Признак | Тип | Описание |
|---------|-----|----------|
| `Major_Category` | текст | Специальность: STEM, Business, Humanities, Medical, Arts |
| `Year_of_Study` | текст | Год обучения: Freshman → Graduate |
| `Pre_Semester_GPA` | число | GPA до семестра (1.0–4.0) |
| `Weekly_GenAI_Hours` | число | Часов в неделю с ИИ-инструментами |
| `Primary_Use_Case` | текст | Для чего использует ИИ |
| `Prompt_Engineering_Skill` | текст | Уровень владения: Beginner / Intermediate / Advanced |
| `Tool_Diversity` | число | Количество разных ИИ-инструментов |
| `Paid_Subscription` | да/нет | Есть ли платная подписка на ИИ |
| `Traditional_Study_Hours` | число | Часов традиционной учёбы в неделю |
| `Perceived_AI_Dependency` | 1–10 | Субъективная зависимость от ИИ |
| `Institutional_Policy` | текст | Политика вуза: запрет / с цитированием / поощрение |
| `Anxiety_Level_During_Exams` | 1–10 | Тревожность на экзаменах |
| `Post_Semester_GPA` | число | GPA после семестра |
| `Skill_Retention_Score` | 0–100 | Удержание навыков |
| `Burnout_Risk_Level` | текст | Риск выгорания: Low / Medium / High |

---

## 2. Загрузка и очистка

```python
df = pd.read_csv('ai_student_impact_dataset.csv')

# Размер датасета
print(df.shape)
# → (50000, 16)

# Проверка пропусков
print(df.isnull().sum())
# → 0 по всем столбцам — данные чистые

# Проверка дубликатов
print(df.duplicated().sum())
# → 0

# Типы данных
print(df.dtypes)
```

```python
# Создаём ключевой признак: прирост GPA за семестр
df['GPA_delta'] = df['Post_Semester_GPA'] - df['Pre_Semester_GPA']

# Группируем часы ИИ в понятные корзины
df['ai_bin'] = pd.cut(
    df['Weekly_GenAI_Hours'],
    bins=[0, 2, 5, 10, 20, 40],
    labels=['0–2 ч', '2–5 ч', '5–10 ч', '10–20 ч', '20+ ч']
)

print(df['GPA_delta'].describe().round(3))
```

**Результат очистки:**
- Строк: **50 000**
- Пропусков: **0**
- Дубликатов: **0**
- Добавлен признак `GPA_delta` — главная метрика анализа

---

## 3. Общая картина

### 3.1 Кто в датасете

```python
# Специальности
print(df['Major_Category'].value_counts())

# Год обучения
print(df['Year_of_Study'].value_counts())

# Политика вуза
print(df['Institutional_Policy'].value_counts())

# Уровень выгорания
print(df['Burnout_Risk_Level'].value_counts())
```

**Специальности:**

| Специальность | Студентов | Доля |
|--------------|----------|------|
| STEM | 15 059 | 30% |
| Business | 12 538 | 25% |
| Humanities | 9 994 | 20% |
| Medical | 6 476 | 13% |
| Arts | 5 933 | 12% |

**Политика вуза:**

| Политика | Студентов |
|---------|----------|
| Разрешено с цитированием | 25 224 (50%) |
| Активно поощряется | 14 988 (30%) |
| Строгий запрет | 9 788 (20%) |

### 3.2 Основная статистика

```python
cols = ['Pre_Semester_GPA', 'Post_Semester_GPA', 'GPA_delta',
        'Weekly_GenAI_Hours', 'Traditional_Study_Hours',
        'Skill_Retention_Score', 'Anxiety_Level_During_Exams']

print(df[cols].describe().round(2))
```

| Показатель | Min | Медиана | Среднее | Max |
|-----------|-----|---------|---------|-----|
| Pre GPA | 1.18 | 3.21 | 3.15 | 4.00 |
| Post GPA | 1.00 | 3.42 | 3.35 | 4.00 |
| **GPA delta** | **-0.92** | **+0.20** | **+0.20** | **+1.01** |
| AI часов/нед | 0 | 5.8 | 8.4 | 40 |
| Трад. часов/нед | 1 | 11.2 | 11.2 | 35.9 |
| Skill Retention | 10.8 | 76.0 | 75.8 | 100 |
| Тревожность | 1 | 4 | 4.3 | 10 |

**Вывод:** в среднем студенты улучшили GPA на **+0.20** за семестр. Есть студенты с падением до -0.92 и ростом до +1.01 — разброс огромный, значит результат сильно зависит от поведения.

### 3.3 Визуализация: распределение изменения GPA

```python
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Гистограмма прироста GPA
axes[0].hist(df['GPA_delta'], bins=60, color='steelblue',
             edgecolor='white', alpha=0.9)
axes[0].axvline(df['GPA_delta'].mean(), color='red', lw=2, ls='--',
                label=f'Среднее: {df["GPA_delta"].mean():.3f}')
axes[0].axvline(0, color='black', lw=1.5, ls=':', label='Нет изменений')
axes[0].set_xlabel('Прирост GPA (Post − Pre)', fontsize=11)
axes[0].set_ylabel('Кол-во студентов', fontsize=11)
axes[0].set_title('Как изменился GPA студентов за семестр?',
                  fontsize=12, fontweight='bold')
axes[0].legend()

# Pie: риск выгорания
burnout_cnt = df['Burnout_Risk_Level'].value_counts()
colors = ['#e74c3c', '#f39c12', '#2ecc71']
axes[1].pie(burnout_cnt, labels=burnout_cnt.index,
            autopct='%1.1f%%', colors=colors, startangle=140)
axes[1].set_title('Распределение риска выгорания',
                  fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('overview.png', bbox_inches='tight')
plt.show()
```

---

## 4. Анализ: часы ИИ vs результат

### 4.1 Как объём использования ИИ связан с GPA и выгоранием

```python
# Считаем метрики по каждой группе часов ИИ
ai_impact = df.groupby('ai_bin', observed=True).agg(
    студентов        = ('Student_ID',          'count'),
    прирост_GPA      = ('GPA_delta',           'mean'),
    burnout_high_pct = ('Burnout_Risk_Level',
                        lambda x: (x == 'High').mean() * 100),
    тревожность      = ('Anxiety_Level_During_Exams', 'mean'),
    skill_retention  = ('Skill_Retention_Score', 'mean')
).round(2)

print(ai_impact)
```

| Часов с ИИ/нед | Студентов | Прирост GPA | Burnout High % | Тревожность | Skill Ret |
|---------------|----------|-------------|----------------|-------------|-----------|
| 0–2 ч | 12 847 | +0.19 | 8.9% | 4.1 | 76.4 |
| 2–5 ч | 10 234 | +0.20 | 11.9% | 4.1 | 76.1 |
| **5–10 ч** | **11 209** | **+0.23** | **19.0%** | **4.2** | **75.8** |
| 10–20 ч | 9 876 | +0.22 | 40.0% | 4.6 | 74.9 |
| **20+ ч** | **5 834** | **+0.16** | **74.3%** | **5.1** | **70.2** |

```python
# График: двойная ось — изменение GPA и выгорание
fig, ax1 = plt.subplots(figsize=(10, 5))
ax2 = ax1.twinx()

x = range(len(ai_impact))
bars = ax1.bar(x, ai_impact['прирост_GPA'],
               color='steelblue', alpha=0.75, label='Прирост GPA')
ax2.plot(x, ai_impact['burnout_high_pct'],
         'ro-', lw=2.5, ms=9, label='Burnout High %')

ax1.set_xticks(x)
ax1.set_xticklabels(ai_impact.index, fontsize=10)
ax1.set_ylabel('Средний прирост GPA', color='steelblue', fontsize=11)
ax2.set_ylabel('Доля с высоким выгоранием (%)', color='red', fontsize=11)
ax1.set_title('Сколько часов ИИ оптимально?', fontsize=13, fontweight='bold')
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

# Выделяем оптимальную зону
ax1.axvspan(1.6, 2.4, alpha=0.08, color='green', label='Оптимум')

plt.tight_layout()
plt.savefig('ai_hours_impact.png', bbox_inches='tight')
plt.show()
```

**Ключевой вывод:**
- **5–10 ч/нед** — оптимальная зона: максимальный прирост GPA (+0.23) при умеренном риске выгорания (19%)
- **20+ ч/нед** — переломная точка: GPA падает до +0.16, и **74% студентов в зоне высокого выгорания**

### 4.2 Традиционная учёба vs прирост GPA

```python
# Группируем традиционные часы учёбы
df['trad_bin'] = pd.cut(df['Traditional_Study_Hours'],
                         bins=[0, 5, 10, 15, 20, 36],
                         labels=['0–5 ч', '5–10 ч', '10–15 ч', '15–20 ч', '20+ ч'])

trad_impact = df.groupby('trad_bin', observed=True)['GPA_delta'].mean().round(3)
print(trad_impact)

plt.figure(figsize=(8, 4))
bars = plt.bar(trad_impact.index, trad_impact.values,
               color=sns.color_palette('Greens_d', 5))
plt.bar_label(bars, fmt='+%.3f', padding=3, fontsize=9)
plt.ylabel('Средний прирост GPA', fontsize=11)
plt.title('Больше традиционной учёбы = лучше GPA?',
          fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()
```

| Трад. часов/нед | Прирост GPA |
|----------------|-------------|
| 0–5 ч | +0.07 |
| 5–10 ч | +0.14 |
| 10–15 ч | +0.21 |
| 15–20 ч | +0.28 |
| **20+ ч** | **+0.34** |

**Вывод:** чёткая линейная зависимость. Традиционная учёба — самый надёжный предиктор роста GPA, сильнее, чем любые параметры использования ИИ.

### 4.3 Корреляционная матрица

```python
num_cols = ['Weekly_GenAI_Hours', 'Traditional_Study_Hours',
            'Perceived_AI_Dependency', 'Anxiety_Level_During_Exams',
            'Skill_Retention_Score', 'GPA_delta', 'Tool_Diversity']

corr = df[num_cols].corr()

plt.figure(figsize=(9, 7))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
            mask=mask, linewidths=0.5, annot_kws={'size': 10},
            vmin=-1, vmax=1)
plt.title('Корреляция между числовыми признаками',
          fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('correlation.png', bbox_inches='tight')
plt.show()
```

**Главные корреляции:**

| Пара признаков | r | Интерпретация |
|----------------|---|---------------|
| AI Hours ↔ AI Dependency | **+0.67** | Больше используешь → сильнее зависишь |
| Traditional Hours ↔ GPA_delta | **+0.37** | Традиционная учёба → лучше результат |
| AI Hours ↔ Anxiety | **+0.27** | Много ИИ → выше тревожность |
| AI Hours ↔ GPA_delta | **-0.05** | Часы с ИИ сами по себе не влияют на GPA |
| Skill Retention ↔ GPA_delta | **+0.20** | Кто лучше удерживает навыки — тот и растёт |

---

## 5. Влияние политики вуза

### 5.1 GPA и выгорание по политике

```python
policy_stats = df.groupby('Institutional_Policy').agg(
    студентов       = ('Student_ID',          'count'),
    средний_GPA     = ('Post_Semester_GPA',   'mean'),
    прирост_GPA     = ('GPA_delta',           'mean'),
    burnout_high    = ('Burnout_Risk_Level',
                       lambda x: (x == 'High').mean() * 100),
    тревожность     = ('Anxiety_Level_During_Exams', 'mean')
).round(3)

print(policy_stats)
```

| Политика | GPA после | Прирост GPA | Burnout High % | Тревожность |
|---------|----------|-------------|----------------|-------------|
| **Активное поощрение** | **3.353** | **+0.206** | **23.8%** | 4.22 |
| Разрешено с цитированием | 3.353 | +0.208 | 23.8% | 4.27 |
| **Строгий запрет** | **3.333** | **+0.187** | **29.8%** | **4.32** |

```python
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Изменение GPA по политике
df.boxplot(column='GPA_delta', by='Institutional_Policy', ax=axes[0],
           grid=False, patch_artist=True,
           boxprops=dict(facecolor='steelblue', alpha=0.6),
           medianprops=dict(color='red', lw=2.5),
           flierprops=dict(marker='o', ms=2, alpha=0.2))
axes[0].set_title('Прирост GPA по политике вуза', fontsize=11, fontweight='bold')
axes[0].set_xlabel('')
axes[0].set_ylabel('GPA delta')
plt.sca(axes[0]); plt.title('Прирост GPA по политике вуза')

# Доля высокого риска выгорания по политике
burnout_policy = df.groupby('Institutional_Policy')['Burnout_Risk_Level'].apply(
    lambda x: (x == 'High').mean() * 100).sort_values()
colors = ['#2ecc71', '#f39c12', '#e74c3c']
bars2 = axes[1].bar(burnout_policy.index, burnout_policy.values, color=colors)
axes[1].bar_label(bars2, fmt='%.1f%%', padding=3, fontsize=10)
axes[1].set_ylabel('Доля студентов с высоким выгоранием (%)', fontsize=10)
axes[1].set_title('Выгорание по политике вуза', fontsize=11, fontweight='bold')
axes[1].tick_params(axis='x', rotation=10)

plt.tight_layout()
plt.savefig('policy_impact.png', bbox_inches='tight')
plt.show()
```

**Вывод:** строгий запрет ИИ — худшая стратегия. У студентов из вузов со Strict_Ban:
- GPA ниже на 0.02 пункта
- Выгорание на **6 п.п. выше** (29.8% vs 23.8%)

---

## 6. Сравнение специальностей

```python
major_stats = df.groupby('Major_Category').agg(
    ai_часов        = ('Weekly_GenAI_Hours',        'mean'),
    трад_часов      = ('Traditional_Study_Hours',   'mean'),
    прирост_GPA     = ('GPA_delta',                 'mean'),
    skill_ret       = ('Skill_Retention_Score',     'mean'),
    burnout_high    = ('Burnout_Risk_Level',
                       lambda x: (x == 'High').mean() * 100)
).round(2).sort_values('прирост_GPA', ascending=False)

print(major_stats)
```

| Специальность | AI ч/нед | Трад. ч/нед | Прирост GPA | Skill Ret | Burnout High % |
|--------------|---------|------------|-------------|-----------|----------------|
| **STEM** | **10.49** | 11.2 | **+0.217** | **76.8** | 25.6% |
| Medical | 7.55 | 11.3 | +0.201 | 75.5 | 24.8% |
| Humanities | 6.77 | 11.1 | +0.198 | 75.3 | 24.7% |
| Arts | 7.27 | 11.1 | +0.197 | 75.7 | 25.1% |
| Business | 8.28 | 11.2 | +0.194 | 75.3 | 24.9% |

```python
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# AI часов
major_stats['ai_часов'].sort_values().plot(
    kind='barh', ax=axes[0], color=sns.color_palette('Blues_d', 5))
axes[0].set_title('Часов с ИИ в неделю', fontsize=11, fontweight='bold')
axes[0].set_xlabel('Среднее часов/нед')

# Прирост GPA
major_stats['прирост_GPA'].sort_values().plot(
    kind='barh', ax=axes[1], color=sns.color_palette('Greens_d', 5))
axes[1].set_title('Прирост GPA', fontsize=11, fontweight='bold')
axes[1].set_xlabel('Средний GPA delta')

# Сохранение навыков
major_stats['skill_ret'].sort_values().plot(
    kind='barh', ax=axes[2], color=sns.color_palette('Oranges_d', 5))
axes[2].set_title('Удержание навыков', fontsize=11, fontweight='bold')
axes[2].set_xlabel('Skill Retention Score')

plt.suptitle('Сравнение специальностей', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('major_comparison.png', bbox_inches='tight')
plt.show()
```

**Вывод:** STEM-студенты используют ИИ больше всех (10.5 ч/нед) и показывают лучший прирост GPA. Причина — они используют ИИ для отладки кода (`Debugging`), что даёт прямой конкретный результат.

---

## 7. Сценарии использования ИИ

```python
use_case_stats = df.groupby('Primary_Use_Case').agg(
    студентов    = ('Student_ID',             'count'),
    прирост_GPA  = ('GPA_delta',              'mean'),
    skill_ret    = ('Skill_Retention_Score',  'mean'),
    тревожность  = ('Anxiety_Level_During_Exams', 'mean')
).sort_values('прирост_GPA', ascending=False).round(3)

print(use_case_stats)
```

| Сценарий | Студентов | Прирост GPA | Skill Ret | Тревожность |
|----------|----------|-------------|-----------|-------------|
| **Debugging/Troubleshooting** | 12 295 | **+0.249** | **78.1** | 4.3 |
| Ideation | 10 721 | +0.200 | 75.5 | 4.2 |
| Copywriting/Drafting | 12 011 | +0.200 | 75.2 | 4.3 |
| Summarizing_Reading | 8 633 | +0.197 | 75.2 | 4.3 |
| **Direct_Answer_Generation** | 6 340 | **+0.133** | **73.7** | **4.3** |

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Прирост GPA
use_case_stats['прирост_GPA'].sort_values().plot(
    kind='barh', ax=axes[0],
    color=sns.color_palette('RdYlGn', 5))
axes[0].set_title('Прирост GPA по сценарию использования ИИ',
                  fontsize=11, fontweight='bold')
axes[0].set_xlabel('Средний GPA delta')

# Сохранение навыков
use_case_stats['skill_ret'].sort_values().plot(
    kind='barh', ax=axes[1],
    color=sns.color_palette('YlGn', 5))
axes[1].set_title('Удержание навыков по сценарию',
                  fontsize=11, fontweight='bold')
axes[1].set_xlabel('Skill Retention Score')

plt.tight_layout()
plt.savefig('use_case_impact.png', bbox_inches='tight')
plt.show()
```

**Критический вывод:**
- **Debugging** → лучший GPA (+0.249) и навыки (78.1). ИИ помогает решать задачи, не заменяя мышление
- **Direct_Answer_Generation** → худший GPA (+0.133) и навыки (73.7). Студенты просто получают готовые ответы — мышление не работает

### 7.2 Навык промпт-инжиниринга

```python
prompt_stats = df.groupby('Prompt_Engineering_Skill').agg(
    прирост_GPA  = ('GPA_delta',             'mean'),
    skill_ret    = ('Skill_Retention_Score', 'mean'),
    тревожность  = ('Anxiety_Level_During_Exams', 'mean')
).round(3)

print(prompt_stats)
```

| Уровень | Прирост GPA | Skill Retention | Тревожность |
|---------|-------------|-----------------|-------------|
| Beginner | +0.185 | 71.1 | 4.3 |
| Intermediate | +0.187 | 75.8 | 4.3 |
| **Advanced** | **+0.248** | **82.1** | 4.2 |

**Разрыв между Beginner и Advanced: +34% по GPA, +15% по навыкам.** Умение работать с ИИ важнее самого факта его использования.

---

## 8. Статистические тесты

### 8.1 T-тест: строгий запрет против активного поощрения

**Вопрос:** статистически значима ли разница в GPA между студентами из вузов с запретом и с поощрением ИИ?

```python
from scipy import stats

ban = df[df['Institutional_Policy'] == 'Strict_Ban']['Post_Semester_GPA']
enc = df[df['Institutional_Policy'] == 'Actively_Encouraged']['Post_Semester_GPA']

t, p = stats.ttest_ind(ban, enc)

print(f"Средний GPA (запрет):     {ban.mean():.4f}")
print(f"Средний GPA (поощрение):  {enc.mean():.4f}")
print(f"Разница:                  {enc.mean() - ban.mean():.4f}")
print(f"t-статистика:             {t:.3f}")
print(f"p-value:                  {p:.4f}")
```

```
Средний GPA (запрет):     3.3327
Средний GPA (поощрение):  3.3534
Разница:                  +0.0207
t-статистика:             -3.197
p-value:                  0.0014
```

**p = 0.0014 < 0.05 → разница статистически значима**

Разница небольшая в абсолютных числах, но при выборке 50 000 человек она устойчива и не случайна.

### 8.2 T-тест: платная подписка и удержание навыков

**Вопрос:** даёт ли платная подписка на ИИ лучшее удержание навыков?

```python
paid = df[df['Paid_Subscription'] == True]['Skill_Retention_Score']
free = df[df['Paid_Subscription'] == False]['Skill_Retention_Score']

t2, p2 = stats.ttest_ind(paid, free)

print(f"Skill Retention (платные): {paid.mean():.3f}")
print(f"Skill Retention (бесплатные): {free.mean():.3f}")
print(f"p-value: {p2:.4f}")

# Сравним также часы использования
print(df.groupby('Paid_Subscription')[['Weekly_GenAI_Hours', 'Skill_Retention_Score']].mean().round(2))
```

```
Skill Retention (платные):    75.43
Skill Retention (бесплатные): 76.07
p-value:                       0.0000
```

| Группа | Skill Retention | AI Часов/нед |
|--------|----------------|--------------|
| Платная | 75.43 | **10.33** |
| Бесплатная | **76.07** | 7.03 |

**p < 0.05 → разница значима, но в пользу бесплатных!**

Платники используют ИИ на 47% больше по времени, но удерживают навыки хуже. Платная подписка → больше использования → выше зависимость → хуже навыки.

### 8.3 Корреляция Спирмена: зависимость от ИИ → тревожность

**Вопрос:** чем сильнее студент зависит от ИИ, тем тревожнее ему на экзамене?

```python
r, p = stats.spearmanr(df['Perceived_AI_Dependency'], df['Anxiety_Level_During_Exams'])
print(f"Коэффициент Спирмена r = {r:.4f}")
print(f"p-value = {p:.6f}")
# → r = 0.2803, p = 0.000000
```

**r = 0.28, p < 0.001** — слабая, но значимая связь.

Студенты, которые субъективно ощущают зависимость от ИИ, сильнее тревожатся на экзаменах. Объяснение простое: «Без ИИ я не справлюсь» → стресс на экзамене, где ИИ недоступен.

### 8.4 Корреляция Спирмена: традиционная учёба → прирост GPA

```python
r2, p2 = stats.spearmanr(df['Traditional_Study_Hours'], df['GPA_delta'])
print(f"r = {r2:.4f}, p = {p2:.6f}")
# → r = 0.3709, p = 0.000000
```

**r = 0.37** — самая сильная корреляция во всём датасете для GPA_delta. Традиционная учёба работает лучше любых ИИ-параметров.

### 8.5 ANOVA: сценарий использования ИИ и изменение GPA

**Вопрос:** значимо ли различается прирост GPA в зависимости от того, для чего студент использует ИИ?

```python
groups = [df[df['Primary_Use_Case'] == uc]['GPA_delta']
          for uc in df['Primary_Use_Case'].unique()]

f, p = stats.f_oneway(*groups)
print(f"F-статистика = {f:.2f}")
print(f"p-value      = {p:.6f}")
# → F = 287.4, p = 0.000000
```

**F = 287, p < 0.001** — среднее изменение GPA различается между сценариями использования ИИ. Это групповая связь, а не доказательство причинного эффекта.

---

## 9. Выводы и рекомендации

### Главные находки

| # | Находка | Доказательство |
|---|---------|---------------|
| 1 | **Оптимум — 5–10 ч/нед с ИИ** | Максимальный прирост GPA (+0.23), умеренное выгорание (19%) |
| 2 | **20+ ч/нед — опасная зона** | Прирост GPA падает, 74% студентов с высоким выгоранием |
| 3 | **Традиционная учёба важнее ИИ** | Корреляция с GPA_delta: r=0.37 — самая высокая в датасете |
| 4 | **При строгом запрете результаты хуже** | В группе Strict_Ban ниже GPA и выше выгорание (29,8% против 23,8%); причинность не установлена |
| 5 | **Direct_Answer — худший сценарий** | GPA delta +0.133 и Skill Retention 73.7 — минимум |
| 6 | **Промпт-инжиниринг реально работает** | Advanced vs Beginner: +34% к GPA delta, +15% к навыкам |
| 7 | **Платная подписка не помогает учиться лучше** | Бесплатные: Skill Retention 76.1 vs платные 75.4 |
| 8 | **Зависимость от ИИ → тревожность** | r = 0.28, p < 0.001 — значимая связь |

### Рекомендации

| Кому | Что делать |
|------|-----------|
| **Университеты** | Проверить переход от Strict_Ban к Allowed_With_Citation контролируемым пилотом; текущий анализ показывает связь, но не оценивает причинный эффект |
| **Университеты** | Протестировать курс по составлению запросов; в данных продвинутые пользователи показывают лучшие результаты, но группы могут различаться по исходной подготовке |
| **Преподаватели** | Отслеживать студентов с 20+ ч/нед ИИ — 74% из них в зоне высокого выгорания |
| **Студенты** | Держаться в зоне 5–10 ч/нед с ИИ — оптимальный баланс результата и здоровья |
| **Студенты** | Использовать готовые ответы осторожно: в этой выборке такой сценарий связан с более низкими GPA и сохранением навыков |
| **Студенты** | Не сокращать традиционные часы учёбы ради ИИ — они важнее для роста GPA |

---

## Приложение: запуск

```bash
pip install pandas numpy scipy matplotlib seaborn
```

```python
# Минимальный стартовый код
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

df = pd.read_csv('ai_student_impact_dataset.csv')
df['GPA_delta'] = df['Post_Semester_GPA'] - df['Pre_Semester_GPA']
```

**Воспроизводимость:** данные без пропусков, все агрегации детерминированы — результаты идентичны при каждом запуске.

---

*Все расчёты получены из опубликованного набора на 50 000 строк. Перед применением выводов нужно проверить происхождение и способ формирования данных.*
