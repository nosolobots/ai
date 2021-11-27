package ai.ch2;

public interface State {
    void enter(BaseGameEntity entity);
    void execute(BaseGameEntity entity);
    void exit(BaseGameEntity entity);
}
