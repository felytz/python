# CMD
## Descarga y extraccion de archivos
Para la descarga de archivos, primero abrimos la terminal, e introducimos el siguiente comando en el directorio que queremos que se descargue nuestro archivo, en nuestro caso en la carpeta cmd, entonces asumiento que se parte del directorio de este LEEME.md(/work) tenemos que:

    cd cmd
    curl -O https://ai.stanford.edu/%7Eamaas/data/sentiment/aclImdb_v1.tar.gz

Y para descomprimir este archivo usamos el comando tar:

    tar -xvf aclImdb_v1.tar.gz

## Exploracion de archivos
Ya descomprimido el archivo, procedemos a explorar los datos, iniciamos cambiando de directorio al de interes:

    cd aclImdb
    cd train

### Preparacion de datos
Nos ineteresa conocer el top y bottom 10, asi que procedemos a conseguir el identificador de las peliculas del archivo de urls, cuya forma es "“http://www.imdb.com/title/tt[#id]/usercomments”, siendo el #id un numero, por lo que nos interesa quitar todo lo que no sea los numeros, lo cual puede ser logrado con el comando “tr -cd [:digit:]”, pero el problema de esto es que tambien nos elimina los espacios, pero solo es cuestion de agregar tambien la excepcion de los espacios, quedando “tr -cd [:digit:][:space:]”, lo cual podemos comprobar (para poder apreciarlo mejor), escribiendo en la terminal “head urls_neg.txt”, apreciando como nomas nos queda los identificadores de las peliculas

Se hace la misma logica para el archivo que contiene todas las reseñas, las cuales tienen la forma “[id]_[calficacion 1-10]”, pero en nuestro caso no es releavante el id, ya que no representa el de las peliculas, por lo que podemos quedarnos solo con la calificacion, teniendo entonces

    ls neg |  sort -n | awk -F '[_.]' '{print $2}'| cat > n_p.txt   # lista de archivos de review negativas, 
                                                                    # se acomodan de menor a mayor(por id), se extrae la califcacion y se guarda
                                                                    # en un archivo de texto
   
    ls pos |  sort -n | awk -F '[_.]' '{print $2}'| cat >> n_p.txt  # lista de archivos de review positivas,  
                                                                    # se acomodan de menor a mayor(por id), se extrae la califcacion y se guarda
                                                                    # en un archivo de texto
     
    
    grep -o 'tt[0-9]\+' urls_pos.txt | sed 's/^tt//' | cat > url.txt  # los id de los títulos en orden
    grep -o 'tt[0-9]\+' urls_pos.txt | sed 's/^tt//' | cat >> url.txt  # los id de los títulos en orden

teniendo asi ahora 2 archivos de textos, uno llamado 'n_p.txt' que tiene las review de las peliculas en orden(neg_pos), y otro llamado url.txt, que tiene las id de las peliculas en orden(neg_pos), por lo que solo faltaria procesar estos datos.

### Proceso de datos
Juntamos los dos archivos usando paste, y procedemos a procesarlos agrupando por id unicos, y obteniendo su promedio de la siguiente forma

    paste url.txt n_p.txt | awk '{id[$1]+=$2; count[$1]++} END {for (i in id) print i, id[i]/count[i]}' | cat>rating.txt

ya teniendo el archivo rating.txt obtenemos los primeros y ultimos 10

    sort -k2,2n rating.txt | head -n 10 | cat>worst_10.txt # archivo de texto con las peores 10 peliculas por id
    sort -k2,2n rating.txt | tail -n 10 | cat>top_10.txt   # archivo de texto con las mejores 10 peliculas por id

para obtener el titulo solamente utilizariamos

    curl -L -A "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3" -s "https://www.imdb.com/title/tt$(awk 'NR==1 {print $1}' top_10.txt)" | grep -o '<title>.*</title>' | sed 's/<\/\?title>//g'

donde NR==1 señala el lugar de la pelicula (del 1 al 10), en este caso 'Keep the River on Your Right: A Modern Cannibal Tale (2000)', y lo demas es comandos necesarios para poder extraer el titulo y no obtener error 403 forbidden, aplicando la misma logica si se quiere saber el titulo de una pelicula en el worst_10

    curl -L -A "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3" -s "https://www.imdb.com/title/tt$(awk 'NR==1 {print $1}' worst_10.txt)" | grep -o '<title>.*</title>' | sed 's/<\/\?title>//g'

obteniendo 'Nana (1926)'.

# Python 
Se utilizo poetry para crear el directorio en el que se trabajó.
## Libreta
La libreta se encuentra en el path work/python/top&bottom_10.ipynb
### Funciones
El path que contiene modulos (work/python/python/module.py) contiene las funciones principales que se utilizaron en la libreta
### Codigo
Es cuestion de correr la libreta, adentro de ella se explican los pasos que se toman
