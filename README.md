
## Docker Environment :whale:

### Consideraciones :construction:

1. Se debe tener instalado docker engine. (https://docs.docker.com/engine/install/) 
2. Se debe tener instalado docker compose. (https://docs.docker.com/compose/install/)

### Ejecución :tractor:

Una vez instalado docker engine y docker compose, en la raiz del proyecto se ejecuta de la siguiente forma:

> [!NOTE] 
> :rotating_light: Al ejecutar este comando por primera vez se realiza la construcción de la imagen

```sh
docker compose -f docker/docker-compose.yaml up
```

> [!NOTE]
> :rotating_light: Si se quiere volver a construir la imagen se utiliza el siguiente comando:

```sh
docker compose -f docker/docker-compose.yaml up --build
```