# TuGraph-OGM

## 简介
TuGraph-OGM(Object Graph Mapping), 源自 `Neo4j-OGM` 项目，TuGraph-OGM
支持将JAVA对象（POJO）映射到TuGraph中，JAVA中的类映射为图中的节点、类中的集合映射为边、类的属性映射为图对象的属性，并提供了对应的函数操作图数据库，因此JAVA开发人员可以在熟悉的生态中轻松地使用TuGraph数据库。同时TuGraph-OGM兼容Neo4j-OGM，Neo4j生态用户可以无缝迁移到TuGraph数据库上。

### TuGraph-OGM功能
TuGraph-OGM提供以下函数操作TuGraph：

| 功能                  | 用法                                                                               |
|---------------------|----------------------------------------------------------------------------------|
| 插入单个节点\边            | void session.save(T object)                                                      |
| 批量插入节点\边            | void session.save(T object)                                                      |
| 删除节点与对应边            | void session.delete(T object)                                                    |
| 删除指定label的全部节点      | void session.deleteAll(Class\<T> type)                                           |
| 清空数据库               | void purgeDatabase()                                                             |
| 更新节点                | void session.save(T newObject)                                                   |
| 根据id查询单个节点          | T load(Class<T> type, ID id)                                                     |
| 根据ids查询多个节点         | Collection\<T> loadAll(Class\<T> type, Collection<ID> ids)                       |
| 根据label查询全部节点       | Collection\<T> loadAll(Class\<T> type)                                           |
| 条件查询                | Collection\<T> loadAll(Class\<T> type, Filters filters)                          |
| Cypher查询（指定返回结果类型）  | T queryForObject(Class\<T> objectType, String cypher, Map<String, ?> parameters) |
| Cypher查询   | Result query(String cypher, Map<String, ?> parameters)                           |


## 编译TuGraph-OGM
```shell
cd tugraph-ogm
mvn clean install -DskipTests
```
## 使用TuGraph-OGM
> 详细示例请参考tugraph-ogm-tests

### 在`pom.xml`中引入依赖

导入Tugraph-ogm

``` 
<dependency>
        <groupId>com.antgroup.tugraph</groupId>
        <artifactId>tugraph-ogm-api</artifactId>
        <version>0.1.0</version>
    </dependency>

    <dependency>
        <groupId>com.antgroup.tugraph</groupId>
        <artifactId>tugraph-ogm-core</artifactId>
        <version>0.1.0</version>
    </dependency>

    <dependency>
        <groupId>com.antgroup.tugraph</groupId>
        <artifactId>tugraph-rpc-driver</artifactId>
        <version>0.1.0</version>
    </dependency>
```

### 构建图对象

构建图对象

```java
@NodeEntity
public class Movie {      // 构建Movie节点
    @Id
    private Long id;      // Movie节点的id
    private String title; // title属性
    private int released; // released属性

    // 构建边ACTS_IN    (actor)-[:ACTS_IN]->(movie)
    @Relationship(type = "ACTS_IN", direction = Relationship.Direction.INCOMING)
    Set<Actor> actors = new HashSet<>();

    public Movie(String title, int year) {
        this.title = title;
        this.released = year;
    }
    
    public Long getId() {
        return id;
    }
    
    public void setReleased(int released) {
        this.released = released;
    }
}

@NodeEntity
public class Actor {      // 构建Actor节点
    @Id
    private Long id;
    private String name;

    @Relationship(type = "ACTS_IN", direction = Relationship.Direction.OUTGOING)
    private Set<Movie> movies = new HashSet<>();

    public Actor(String name) {
        this.name = name;
    }

    public void actsIn(Movie movie) {
        movies.add(movie);
        movie.getActors().add(this);
    }
}
```
### 与TuGraph建立连接

使用Tugraph-ogm 建立连接


```java
// 配置
String databaseUri = "list://ip:port";
String username = "admin";
String password = "password";
//启动driver
Driver driver = new RpcDriver();
Configuration.Builder baseConfigurationBuilder = new Configuration.Builder()
                            .uri(databaseUri)
                            .verifyConnection(true)
                            .credentials(username, password);
                            driver.configure(baseConfigurationBuilder.build()); 
driver.configure(baseConfigurationBuilder.build());
// 开启session
SessionFactory sessionFactory = new SessionFactory(driver, "entity_path");
Session session = sessionFactory.openSession();
```

### 通过OGM进行增删改查

使用Tugraph-ogm 进行增删改查

```java
// 增
Movie jokes = new Movie("Jokes", 1990);  // 新建Movie节点jokes
session.save(jokes);                     // 将jokes存储在TuGraph中

Movie speed = new Movie("Speed", 2019);
Actor alice = new Actor("Alice Neeves");
alice.actsIn(speed);                    // 将speed节点与alice节点通过ACTS_IN进行连接
session.save(speed);                    // 存储两个节点与一条边
        
// 删
session.delete(alice);                  // 删除alice节点以及相连的边
Movie m = session.load(Movie.class, jokes.getId());   // 根据jokes节点的id获取jokes节点
session.delete(m);                                    // 删除jokes节点
        
// 改
speed.setReleased(2018);
session.save(speed);                   // 更新speed节点属性
        
// 查  
Collection<Movie> movies = session.loadAll(Movie.class);  // 获取所有Movie节点
Collection<Movie> moviesFilter = session.loadAll(Movie.class,
        new Filter("released", ComparisonOperator.LESS_THAN, 1995));  // 查询所有小于1995年发布的电影
        
// 调用Cypher
HashMap<String, Object> parameters = new HashMap<>();
parameters.put("Speed", 2018);
Movie cm = session.queryForObject(Movie.class,
        "MATCH (cm:Movie{Speed: $Speed}) RETURN *", parameters);      // 查询Speed为2018的Movie

session.query("CALL db.createVertexLabel('Director', 'name', 'name'," +
        "STRING, false, 'age', INT16, true)", emptyMap());            // 创建节点Label Director
session.query("CALL db.createEdgeLabel('DIRECT', '[]')", emptyMap()); // 创建边Label DIRECT
Result createResult = session.query(
        "CREATE (n:Movie{title:\"The Shawshank Redemption\", released:1994})" +
        "<-[r:DIRECT]-" +
        "(n2:Director{name:\"Frank Darabont\", age:63})", 
        emptyMap());
QueryStatistics statistics = createResult.queryStatistics();          // 获取create结果
System.out.println("created " + statistics.getNodesCreated() + " vertices");    // 查看创建节点数目
System.out.println("created " + statistics.getRelationshipsCreated() + " edges");  //查看创建边数目
        
// 清空数据库
session.deleteAll(Movie.class);        // 删除所有Movie节点
session.purgeDatabase();               // 删除全部数据
```
