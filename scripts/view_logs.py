import os
import sys
import click
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from tabulate import tabulate

# Añadir el directorio raíz al path para poder importar desde src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.logger import AccessLogger

@click.group()
def cli():
    """Herramienta para visualizar y analizar los registros de acceso"""
    pass

@cli.command()
@click.option('--name', help='Filtrar por nombre')
@click.option('--days', type=int, default=7, help='Número de días a mostrar')
@click.option('--access-type', type=click.Choice(['PERMITIDO', 'DENEGADO']), help='Tipo de acceso')
def list(name, days, access_type):
    """Lista los registros de acceso"""
    logger = AccessLogger()
    
    # Calcular fecha de inicio
    start_date = datetime.now() - timedelta(days=days)
    
    # Obtener registros
    records = logger.get_access_history(
        name=name,
        start_date=start_date,
        access_type=access_type
    )
    
    if not records:
        print("No se encontraron registros que coincidan con los criterios.")
        return
    
    # Convertir a DataFrame para mejor visualización
    df = pd.DataFrame(records)
    
    # Mostrar resultados
    print(f"\nRegistros de acceso (últimos {days} días):")
    print(tabulate(df, headers='keys', tablefmt='psql', showindex=False))
    print(f"\nTotal de registros: {len(records)}")

@cli.command()
@click.option('--days', type=int, default=7, help='Número de días a analizar')
@click.option('--output', help='Ruta para guardar el gráfico')
def stats(days, output):
    """Muestra estadísticas de acceso"""
    logger = AccessLogger()
    
    # Calcular fecha de inicio
    start_date = datetime.now() - timedelta(days=days)
    
    # Obtener registros
    records = logger.get_access_history(start_date=start_date)
    
    if not records:
        print("No se encontraron registros para el período especificado.")
        return
    
    # Convertir a DataFrame
    df = pd.DataFrame(records)
    
    # Convertir timestamp a datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Estadísticas por día
    df['fecha'] = df['timestamp'].dt.date
    daily_stats = df.groupby(['fecha', 'acceso']).size().unstack(fill_value=0)
    
    # Estadísticas por usuario
    user_stats = df.groupby(['nombre', 'acceso']).size().unstack(fill_value=0)
    
    # Mostrar estadísticas
    print("\nEstadísticas de acceso por día:")
    print(tabulate(daily_stats, headers='keys', tablefmt='psql'))
    
    print("\nEstadísticas de acceso por usuario:")
    print(tabulate(user_stats, headers='keys', tablefmt='psql'))
    
    # Crear gráficos
    plt.figure(figsize=(12, 10))
    
    # Gráfico de accesos por día
    plt.subplot(2, 1, 1)
    daily_stats.plot(kind='bar', ax=plt.gca())
    plt.title('Accesos por día')
    plt.xlabel('Fecha')
    plt.ylabel('Número de accesos')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Gráfico de accesos por usuario
    plt.subplot(2, 1, 2)
    user_stats.plot(kind='barh', ax=plt.gca())
    plt.title('Accesos por usuario')
    plt.xlabel('Número de accesos')
    plt.ylabel('Usuario')
    plt.tight_layout()
    
    # Guardar o mostrar
    if output:
        plt.savefig(output)
        print(f"\nGráfico guardado en: {output}")
    else:
        plt.show()

@cli.command()
@click.option('--format', 'format_type', type=click.Choice(['csv', 'json']), default='csv', help='Formato del reporte')
@click.option('--output', help='Ruta del archivo de salida')
def report(format_type, output):
    """Genera un reporte de accesos"""
    logger = AccessLogger()
    report_path = logger.generate_report(output_file=output, format_type=format_type)
    
    if report_path:
        print(f"Reporte generado exitosamente: {report_path}")
    else:
        print("Error al generar el reporte.")

if __name__ == '__main__':
    cli()