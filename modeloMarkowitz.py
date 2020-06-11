import numpy as np
import pandas as pd

import cvxopt as opt
from cvxopt import solvers #, blas

from matplotlib import pyplot as plt
plt.style.use('seaborn')
np.random.seed(9062020)


# Cargar y limpiar datos
df = pd.read_csv("stocks.csv", sep=",", engine="python")
#���Field 1 la columna tiene caracteres extranios
df.columns = ['Field1' if i == 0 else x for i, x in enumerate(df.columns)]
df.set_index(df.columns[0], inplace=True)

# Seleccionamos 5 activos al azar
activos = np.random.permutation(df.columns)[:5]
plt.figure(figsize=(8,6))
plt.title('Historico de Algunos Activos', color='gray')
for activo in activos:
    plt.plot(df[activo].to_numpy(), label=activo)
plt.ylabel('Precio del Activo', color='gray')
plt.xlabel('Observaciones', color='gray')
plt.legend(loc='upper left')

# Por que usar rendimiento logaritmicos
# https://quantdare.com/por-que-usar-rendimientos-logaritmicos/
df = df / df.shift(1) 
df.dropna(inplace=True)
log_df = np.log(df)
print(log_df)


# Grafico con Rendimientos historicos
plt.figure(figsize=(8,6))
plt.title('Rendimiento de Todos los Activos', color='gray')
plt.plot(df.to_numpy(), alpha=0.2)
plt.ylabel('Rendimiento del Activo', color='gray')
plt.xlabel('Observaciones', color='gray')



def metricas_historicas_portafolio(portafolio, dias_anual):
    """
        Da un pequenio reporte sobre las observaciones de
        cada activo contenido en el portafolio, para 
        devolver finalmente dos matrices

        `portafolio`: es un dataframe de observaciones (filas)
        por activos (columnas) historico

        `dias_anual`: es un entero que indica cuantos dias
        tiene el anio, asumiendose para ellos que las
        observaciones contenidas en `portafolio` son diarias

        `return`: devuelve las matrices o arrays de rendimientos
        esperados y la de covarianza del portafolio historico,
        asi como el numero de activos
    """

    # Metricas Historicas
    activos_en_portafolio = portafolio.shape[1]
    rendimientos_anual = dias_anual * portafolio.mean()
    sigma_de_rendimientos_anual = dias_anual * portafolio.std()
    varianza_de_rendimientos_anual = dias_anual * portafolio.std() ** 2  # diagonal de la covarianza de rendimientos
    covarianza_de_rendimientos_anual = dias_anual * portafolio.cov()

    # Reporte de Historicos
    print(f"\nNumero de Activos:\n{activos_en_portafolio}")
    print(f"\nRendimiento:\n{rendimientos_anual}")
    print(f"\nDesviacion Estandar:\n{sigma_de_rendimientos_anual}")
    print(f"\nVarianza:\n{varianza_de_rendimientos_anual}")
    print(f"\nCovarianza:\n{covarianza_de_rendimientos_anual}")

    # Matrices con las Estadisticas Hitoricas de Interes
    # la variable `C`es mayuscula 

    p = np.asmatrix(rendimientos_anual.to_numpy())
    C = np.asmatrix(covarianza_de_rendimientos_anual.to_numpy())

    return p, C, activos_en_portafolio


p, C, numero_de_activos = metricas_historicas_portafolio(log_df, 252)
#print(p ,p.shape[1])

def resultados_portafolio(p,w,C):
    """
        Dados unos pesos de colocacion para un
        portafolio y teniendose los rendimientos y
        covarianzas historicas, se obtiene el
        rendimiento y volatilidad del portafolio

        `p`: matriz con rendimientos historicos del
        portafolio

        `w`: peso que se empleara para colocar los
        los fondos en los activos correspondientes
        del portafolio

       `C`: matriz con la covarianza historico del
        portafolio

        `return`: el redimiento y riesgo (volatilida)
        del portafolio
    """
    mu = w * p.T                           # Rendimiento Esperado
    sigma = np.sqrt(w * C * w.T)           # Volatilidad

    return mu, sigma

    

def simular_pesos_portafolio(numero_de_activos):
    """ 
        Generar pesos aleatorios para cada
        activo en el portafolio

        `numero_de_activos`: es entero

        `return`:El peso de cada uno de los 
         activos en el portafolio cuya suma es 1
         como matriz
    """
    pesos = np.random.random(numero_de_activos)
    pesos *= sum([np.random.binomial(1, 0.08, numero_de_activos) for _ in range(2)])
    pesos = np.asmatrix(pesos / sum(pesos) )

    return pesos



def simular_portafolio(p, C, numero_de_activos, libre_riesgo=0, limite_volatilidad=1):
    """
        Genera el redimiento y la desviacion estandar
        de una posible combinacion en la inversion
        de cada activo para un portafolio dado

        `p`: matriz con rendimientos historicos del
        portafolio

       `C`: matriz con la covarianza historico del
        portafolio 

        `numero_de_activos`: entero que indica la cantidad
        de activos en el portafolio

        `libre de riesgo`: flotante que va de 0 a 1

        `limite_volatilidad`: es para mantener la
        volatilidad hasta un tope durante la
        simulacion

        `return`: el peso de inversion, el rendimiento
        esperado (mu) y la desviacion estandar (sigma)
        tambien conocida como volatilidad para el 
        portafolio generado así como el Sharpe Ratio
        todas las salidas son arrays
    """
    
    # Generar una posible combinacion del portafolio
    p = p
    w = simular_pesos_portafolio(numero_de_activos)
    C = C

    mu, sigma = resultados_portafolio(p,w,C)

    
    sharpe_ratio = (mu - libre_riesgo) / sigma
    # Esta recursividad reduce los valores atípicos
    # para mantener el portafolio de interés
    # tambien se puede desarrollar con `while` pero
    # se requiere más codigo
    if sigma > limite_volatilidad:
        return simular_portafolio(p, C, numero_de_activos, libre_riesgo, limite_volatilidad)
    
    return w, mu, sigma, sharpe_ratio

peso_activos, rendimiento, volatilidad, sharpe_ratio = simular_portafolio(p, C, numero_de_activos)

print("-"*40)
print('---- Portafolio Simulado ----')
print("-"*40)
print(f"\nSharpe Ratio: {sharpe_ratio}")
print(f"""
Pesos Del Portafolio Simulado:\n{peso_activos}
 \nLos Pesos Suman: {peso_activos.sum():.4f}
""")
print(f"\nRedimiento del Portafolio Simulado:{rendimiento}")
print(f"\nVolatilidad del Portafolio Simulado:{volatilidad}\n")



def simulacion_de_portafolios(numero_de_portafolios, p, C,
    numero_de_activos, libre_riesgo=0, limite_volatilidad=1):

    """
        Genera los rendimientos y volatidades para un conjunto
        de portafolios

        `numero_de_portafolios`: entero que indica la
        cantidad de replicas o simulaciones a efectuarse

        `p`: matriz con rendimientos historicos del
        portafolio

       `C`: matriz con la covarianza historico del
        portafolio 

        `numero_de_activos`: entero que indica la cantidad
        de activos en el portafolio

        `libre de riesgo`: flotante que va de 0 a 1

        `limite_volatilidad`: es para mantener la
        volatilidad hasta un tope durante la
        simulacion

        `return`: los pesos, rendimientos esperados así
        como las volatidades `desviacion estandar`
        para cada uno de los portafolios simulados
    """

    pesos, rendimientos, volatilidades, sharper_ratios = zip(*[
        simular_portafolio(p, C, numero_de_activos, libre_riesgo, limite_volatilidad)
        for _ in range(numero_de_portafolios)
    ])

    pesos, rendimientos, volatilidades, sharper_ratios = \
    np.array(pesos), np.array(rendimientos), np.array(volatilidades), np.array(sharper_ratios)
    
    return pesos, rendimientos, volatilidades, sharper_ratios



pesos, rendimientos, volatilidades, sharper_ratios = simulacion_de_portafolios(
    numero_de_portafolios=1000,
    p=p,
    C=C,
    numero_de_activos=numero_de_activos,
    libre_riesgo=0
)

# Metricas Sharper
def rsharpe_maximo(sharper_ratios, rendimientos, volatilidades):
    maximo_sharpe_ratio = sharper_ratios.max()
    indice_maximo_sharpe_ratio = sharper_ratios.argmax()
    pesos_optimos_simulados = pesos[indice_maximo_sharpe_ratio, :]
    maximo_sharpe_ratio_rendimiento = rendimientos[indice_maximo_sharpe_ratio]
    maximo_sharpe_ratio_volatilidad = volatilidades[indice_maximo_sharpe_ratio]

    print("-" * 50)

    print('---- Estadisticas Sharper Ratio ----')
    print("-" * 50)
    
    print(f"\nMaximo Sharpe Ratio: {maximo_sharpe_ratio}")

    print(f"""Pesos Del Portafolio:\n{pesos_optimos_simulados}
    \nLos Pesos Suman: {pesos_optimos_simulados.sum():.4f}
    """ )
    
    print(f"\nRedimiento del Maximo Sharpe Ratio:{maximo_sharpe_ratio_rendimiento}")
    print(f"\nVolatilidad del Maximo Sharpe Ratio:{maximo_sharpe_ratio_volatilidad}\n")

    return maximo_sharpe_ratio, maximo_sharpe_ratio_volatilidad, maximo_sharpe_ratio_rendimiento


# Estadisticas de Montecarlo
maximo_sharpe_ratio, maximo_sharpe_ratio_volatilidad, maximo_sharpe_ratio_rendimiento = rsharpe_maximo(sharper_ratios, rendimientos, volatilidades)



plt.figure(figsize=(8,6))
plt.title('Rendimientos y Volatilidades\n Portafolios Simulados', color='gray')
plt.scatter(volatilidades, rendimientos, c=sharper_ratios, cmap='cool')
plt.colorbar(label=r"$Sharpe\ Ratio$")

# Optimo Sharpe Ratio Simulado
plt.scatter(
    maximo_sharpe_ratio_volatilidad,
    maximo_sharpe_ratio_rendimiento,
    c='orange', s=60, edgecolors='gray', label=f'Sharpe Ratio Optimo Simulado = {maximo_sharpe_ratio:.4f}'
)

plt.ylabel(r'$Rendimiento$', color='gray')
plt.xlabel(r'$Volatilidad\ \sigma$', color='gray')
plt.legend(loc="upper left")




# Resolviendo el modelo cuadratico
# http://cvxopt.org/userguide/coneprog.html
# http://cvxopt.org/examples/book/portfolio.html
# http://cvxopt.org/examples/tutorial/qp.html
def portafolio_optimo(p, C, numero_de_activos):
    """
        Genera los puntos para la Frontera Eficiente
        `p`: matriz con rendimientos historicos del
        portafolio

       `C`: matriz con la covarianza historico del
        portafolio 

        `numero_de_activos`: entero que indica la cantidad
        de activos en el portafolio

        `retorna`: arrays, de los pesos de cada portafolio 
        correspondientes a cada punto de la frontera 
        eficiente, siendo dichos puntos el par rendimiento
        y volatilidad 
    """

    # Se establece saltos discretos para hallar la
    # la frontera eficiente estos seran los 
    # `targets` u objetivos que se fijan para optimizar
    N = 100
    n = numero_de_activos
    #mus = np.power(10, 5 * np.arange(N) / N - 1) # tiene que ser lista tolist()
    mus = [ 10**(5.0*t/N-1.0) for t in range(N) ]
    
    # convertir el p y C a matrices del tipo cvxopt
    # en el caso de p se trabaja con su transpuesta
    pbar = opt.matrix(p.T)
    S = opt.matrix(C)

    # Crear las matrices de restricciones
    # Gx <= h
    G = -opt.matrix(np.eye(n))   # matriz identidad negativa n x n 
    h = opt.matrix(0.0, (n ,1))

    # Ax = b
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)         # La suma de los pesos es 1


    # Calcular los pesos de la frontera eficiente
    # Empleando Programacion Cuadratica
    # Pero primero silenciamos el solver (es opcional)
    solvers.options['show_progress'] = False
    portafolios = [solvers.qp(mu*S, -pbar, G, h, A, b)['x'] for mu in mus]

    # Calcular los rendimientos y volatilidades o riesgos
    # para la frontera eficiente
    # Estas implementaciones funcionan... hay que importar "blas"
    # pero el codigo que esta fuera de este codigo
    # requiere que se redimensionen los rendimientos, volatilidades
    # para que funcionen
    #rendimientos = [blas.dot(pbar, x) for x in portafolios]
    #volatilidades = [np.sqrt(blas.dot(x, S*x)) for x in portafolios]

    rendimientos, volatilidades = zip(*[
        resultados_portafolio(p, np.array(w).T, C) for w in portafolios
    ])


    # Calcular el portafolio optimo
    #pesos = solvers.qp(opt.matrix(x1 * S), -pbar, G, h, A, b)['x'] # # No funciona para lo que se quiere hacer mas adelante
    pesos = [np.asarray(x) for x in portafolios]

    pesos = np.asarray(pesos)
    rendimientos = np.asarray(rendimientos)
    volatilidades = np.asarray(volatilidades)

    return pesos, rendimientos, volatilidades


w_optimos, mu_optimos, sigma_optimos = portafolio_optimo(p, C, numero_de_activos)



########################################
# Frontera Eficiente y Simulacion
########################################
plt.figure(figsize=(8,6))
plt.title('Frontera eficiente', color='gray')

# Frontera eficiente
plt.plot(sigma_optimos.reshape((1,-1))[0], mu_optimos.reshape((1,-1))[0], 'y-o', color='gray', alpha=0.4, label='Frontera Eficiente')

# Simulados
plt.scatter(volatilidades, rendimientos, c=sharper_ratios, cmap='cool')
plt.colorbar(label=r"$Sharpe\ Ratio$")

# Optimo Sharpe Ratio Simulado
plt.scatter(
    maximo_sharpe_ratio_volatilidad,
    maximo_sharpe_ratio_rendimiento,
    c='orange', s=60, edgecolors='gray', label=f'Sharpe Ratio Optimo Simulado = {maximo_sharpe_ratio:.4f}'
)

plt.ylabel(r'$Rendimiento$', color='gray')
plt.xlabel(r'$Volatilidad\ \sigma$', color='gray')
plt.legend(loc="upper left")



########################################
# Ratio de Sharper
########################################
print("\nVerificar cuales suman 1: ")
print(np.array([x.sum() for x in w_optimos]))

filtrar_pesos_positivos = np.array([(x>=0).all() for x in w_optimos])
print("\nVerificar que todos los pesos sean >= 0: ")
print(filtrar_pesos_positivos)
print(w_optimos.shape, mu_optimos.shape, sigma_optimos.shape)


w_optimos = w_optimos[filtrar_pesos_positivos]
mu_optimos = mu_optimos [filtrar_pesos_positivos]
sigma_optimos = sigma_optimos[filtrar_pesos_positivos]


print("\nVerificar que todos los pesos sean >= 0: ")
print(np.array([(x>=0).all() for x in w_optimos]))
print(w_optimos.shape, mu_optimos.shape, sigma_optimos.shape)


libre_riesgo = 0
rsharpe_optimos = (mu_optimos - libre_riesgo) / sigma_optimos
rsharpe_optimos = rsharpe_optimos.reshape((1,-1)).reshape((1,-1)) # quitarle dimensiones


maximo_rsharpe, maximo_rsharpe_volatilidad, maximo_rsharpe_rendimiento = rsharpe_maximo(rsharpe_optimos, mu_optimos, sigma_optimos)


plt.figure(figsize=(8,6))
plt.title('Sharpers de la Frontera Eficiente\nCon Pesos no Negativos', color='gray')

# Optimo Sharpe Ratio Frontera Eficiente
plt.scatter(
    rsharpe_optimos[0].argmax(),
    maximo_rsharpe,
    c='#90ff1e', s=100, edgecolors='gray', label=f'Sharpe Ratio = {maximo_rsharpe:.4f}'
)

plt.plot(rsharpe_optimos[0], 'y-o', color='dodgerblue', alpha=0.4)

plt.ylabel(r'$Sharpe\ Ratio$', color='gray')
plt.xlabel(r'$Observaciones$', color='gray')
plt.legend(loc="upper left")




########################################
# Todo Junto:
# Ratio de Sharper Optimo
# Frontera Eficiente y Simulacion
########################################
w_optimos, mu_optimos, sigma_optimos = portafolio_optimo(p, C, numero_de_activos)
libre_riesgo = 0
rsharpe_optimos = (mu_optimos - libre_riesgo) / sigma_optimos
rsharpe_optimos = rsharpe_optimos.reshape((1,-1)).reshape((1,-1)) # quitarle dimensiones



plt.figure(figsize=(8,6))
plt.title('Frontera eficiente', color='gray')

# Frontera eficiente
plt.plot(sigma_optimos.reshape((1,-1))[0], mu_optimos.reshape((1,-1))[0], 'y-o', color='gray', alpha=0.4, label='Frontera Eficiente')

# Simulados
plt.scatter(volatilidades, rendimientos, c=sharper_ratios, cmap='cool')
plt.colorbar(label=r"$Sharpe\ Ratio$")

# Optimo Sharpe Ratio Frontera Eficiente
idx_rshape_optimo = np.where(rsharpe_optimos[0] == maximo_rsharpe)
plt.scatter(
    sigma_optimos.reshape((1, -1))[0][idx_rshape_optimo],   # eje volatilidad
    mu_optimos.reshape((1,-1))[0][idx_rshape_optimo],       # eje rendimientos
    c='#90ff1e', s=100, edgecolors='gray', label=f'Sharpe Ratio Optimo = {maximo_rsharpe:.4f}'
)

# Optimo Sharpe Ratio Simulado
plt.scatter(
    maximo_sharpe_ratio_volatilidad,
    maximo_sharpe_ratio_rendimiento,
    c='orange', s=100, edgecolors='gray', label=f'Sharpe Ratio Optimo Simulado = {maximo_sharpe_ratio:.4f}'
)

plt.ylabel(r'$Rendimiento$', color='gray')
plt.xlabel(r'$Volatilidad\ \sigma$', color='gray')
plt.legend(loc="upper left")



########################################
# Se muestran todos los lienzos
########################################
plt.show()
