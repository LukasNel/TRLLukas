import asyncio

async def async_gen_with_send():
    value = yield "start"
    while True:
        value = yield f"received: {value}"

async def test_async_gen_send():
    gen = async_gen_with_send()
    
    # First yield returns "start"
    result = await gen.asend(None)
    print(f"First yield: {result}")
    assert result == "start"
    
    # Send "hello" and get response
    result = await gen.asend("hello")
    print(f"After sending 'hello': {result}")
    assert result == "received: hello"
    
    # Send "world" and get response
    result = await gen.asend("world")
    print(f"After sending 'world': {result}")
    assert result == "received: world"
    
    # Close the generator
    await gen.aclose()
    print("Generator closed successfully")

async def test_async_gen_send_error():
    gen = async_gen_with_send()
    
    # First yield returns "start"
    result = await gen.asend(None)
    print(f"First yield: {result}")
    assert result == "start"
    
    # Try to send to a closed generator
    await gen.aclose()
    print("Generator closed")
    
    try:
        await gen.asend("test")
        print("Error: Should have raised StopAsyncIteration")
    except StopAsyncIteration:
        print("Successfully caught StopAsyncIteration")

async def main():
    print("Testing normal send behavior:")
    await test_async_gen_send()
    print("\nTesting error case:")
    await test_async_gen_send_error()

if __name__ == "__main__":
    asyncio.run(main())
