package ai.ch2;

public abstract class BaseGameEntity {
    private int id;
    private String name;
    State currentState;
    Location location;

    public BaseGameEntity(int id, String name) {
        this.id = id;
        this.name = name;
    }

    public int getId() { return this.id; }
    public String getName() { return this.name; }

    public void changeState(State newState) {
        currentState.exit(this);
        this.currentState = newState;
        newState.enter(this);
    }

    public void changeLocation(Location newLocation) { this.location = newLocation; }
    public abstract void update();
}
