# from unittest import result
import streamlit as st
import numpy as np
from tsp_branch_and_bound import (
    format_matrix,
    tsp_branch_and_bound,
    draw_graph,
    format_matrix,
    generate_letters,
)
import pandas as pd

st.title("Задача комівояжера: Гілки та межі")
st.write("Введіть матрицю відстаней для вирішення задачі комівояжера.")

size = st.number_input("Введіть розмір матриці:", min_value=2, max_value=10, value=5)
nodes = generate_letters(size)
initial = pd.DataFrame(np.zeros(shape=(size, size)), index=nodes, columns=nodes)
edited_df = st.data_editor(initial)
matrix = edited_df.replace([np.inf, -np.inf], "inf").to_numpy()

if st.button("Обчислити оптимальний шлях"):
    best_cost, best_path, results = tsp_branch_and_bound(matrix)
    st.write(f"Найменша вартість: {best_cost}")
    st.write(f"Найкращий шлях: {" -> ".join([nodes[i] for i in best_path])}")

    st.write("Таблиця кроків")
    for result in results:
        st.markdown(f"# Рівень {result['Level']}")
        st.write(f"Шлях {" -> ".join([nodes[i] for i in result['Path']])}")
        st.write(f"Вартість шляху {result['Path cost']}")
        st.write("Знижена матриця")
        df = pd.DataFrame(result["Reduced Matrix"])
        st.dataframe(format_matrix(result["Reduced Matrix"]))
        st.write("")

    st.write("Граф з оптимальним шляхом:")
    st.pyplot(draw_graph(matrix, best_path))
